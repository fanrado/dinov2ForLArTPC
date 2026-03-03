# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection

from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger


logger = logging.getLogger("dinov2")


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, batch_size, num_workers, gather_on_cpu=False):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels

###================ Evaluation of the features extracted by the dinov2 model (without running a supervised head on top of the features) ==================###
class eval_dino_features:
    def __init__(self, dino_model, n_last_blocks, dataloader, str_labels: list[str] = None):
        self.dino_model = dino_model
        self.n_last_blocks = n_last_blocks
        self.dataloader = dataloader
        self.str_labels = str_labels
    
    # @torch.inference_mode()
    # def extract_features(self, images):
    #     with torch.inference_mode():
    #         features = self.dino_model.get_intermediate_layers(
    #             images, self.n_last_blocks, return_class_token=True
    #         )
    #     return features
    @torch.inference_mode()
    def extract_features(self, images):
        return self.dino_model(images)
    
    def _to_feature_matrix(self, features):
        """Convert DINO intermediate-layer output into a [B, D] tensor (uses CLS token)."""
        if isinstance(features, (list, tuple)):
            features = features[-1]                      # last block

        if isinstance(features, (list, tuple)) and len(features) == 2:
            print('Extracting CLS token from features returned as (patch_tokens, cls_token)======')
            _, cls_token = features                      # (patch_tokens, cls_token)
            return cls_token.float()

        if torch.is_tensor(features):
            if features.dim() == 3:                      # [B, N, D] – take CLS position
                return features[:, 0, :].float()
            if features.dim() == 2:                      # already [B, D]
                return features.float()

        raise RuntimeError("Unsupported feature format from get_intermediate_layers")

    @torch.inference_mode()
    def extract_all_features(self):
        '''
            Extract features for all samples in the dataloader, gather them across processes, and return as a single feature matrix and label vector.
        '''
        all_features = []
        all_labels = []
        for samples, targets in self.dataloader:
            features = self.extract_features(samples.cuda(non_blocking=True))
            features = self._to_feature_matrix(features).cpu()

            labels = targets[1] if isinstance(targets, (tuple, list)) and len(targets) == 2 else targets
            labels = labels.cpu()

            all_features.append(features)
            all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0)   # [N, D]
        all_labels = torch.cat(all_labels, dim=0)        # [N]
        return all_features, all_labels

    def calculate_tSNE(self, features, labels, n_components=2, perplexity=30.0):
        """
            Calculate the t-SNE embedding of the features and return the 2D coordinates along with the corresponding labels:
            Arguments:
                features: [N, D] tensor of features
                labels: [N] tensor of labels
                n_components: number of dimensions for t-SNE (default 2)
                perplexity: t-SNE perplexity parameter (default 30.0) which is the effective number of neighbors considered for each point. Adjust based on dataset size (e.g., 5-50).
            Returns:
                tsne_coords: [N, n_components] tensor of t-SNE coordinates
                labels: [N] tensor of labels (same as input)
        """
        from sklearn.manifold import TSNE
        import numpy as np

        features_np = features.cpu().numpy()
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_coords_np = tsne.fit_transform(features_np)
        tsne_coords = torch.from_numpy(tsne_coords_np).float()
        return tsne_coords, labels
    
    def visualize_tSNE(self, tsne_coords, labels):
        """
            This function visualizes the t-SNE coordinates in a 2D scatter plot, coloring points by their labels. This shows us how well the DINO features cluster according to the labels, which can indicate how well the self-supervised features capture class structure.
            Arguments:
                tsne_coords: [N, 2] tensor of t-SNE coordinates
                labels: [N] tensor of labels corresponding to each point
            Returns:
                A matplotlib figure object containing the t-SNE scatter plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        tsne_coords_np = tsne_coords.cpu().numpy()
        labels_np = labels.cpu().numpy()

        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(tsne_coords_np[:, 0], tsne_coords_np[:, 1], c=labels_np, cmap='tab10', alpha=0.7)
        if self.str_labels is not None:
            legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
            plt.gca().add_artist(legend1)
            plt.legend(handles=scatter.legend_elements()[0], labels=self.str_labels, title="Classes", loc='upper right')
        plt.title('t-SNE Visualization of DINO Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()
    
    def run_evaluation(self):
        features, labels = self.extract_all_features()
        tsne_coords, labels = self.calculate_tSNE(features, labels)
        fig = self.visualize_tSNE(tsne_coords, labels)
        return fig

from linear_cvn import load_yaml_config, prepare_dataset
import click
from linear_cvn import load_dinov2_backbone, load_config_pretraining
@click.command()
@click.option('--model-config', required=True, type=click.Path(exists=True),
              help='Path to the model configuration YAML file (config_dict).')
@click.option('--data-config', required=True, type=click.Path(exists=True),
              help='Path to the data configuration YAML file (params_dict).')
@click.option('--device', default=None, type=str,
              help='Device to use for training (e.g. cuda:0, cpu). Defaults to cuda:0 if available.')
def main(model_config, data_config, device):
    '''
        Main entry point for linear evaluation of DINOv2 backbone on CVN data.
        Loads model and data configurations from YAML files, prepares the dataset,
        trains the linear classifier, and runs evaluation on the test set.
    '''
    # Load configurations from YAML files
    config_dict = load_yaml_config(model_config)
    params_dict = load_yaml_config(data_config)

    # Extract the pretraining config path from the model config
    PATH_TO_CONFIG = config_dict.pop('pretraining_config_path')

    # Set device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'Using device: {device}')

    print(f'Training for : {params_dict["classification_type"]} classification')
    dataloaders, class_names, datasets_test, dataset_splitting = prepare_dataset(
        params_dict=params_dict, img_size=config_dict['img_size']
    )

    config_pretraining = load_config_pretraining(PATH_TO_CONFIG)

    feature_model, embed_dim = load_dinov2_backbone(
        config_dict=config_dict, config_pretraining=config_pretraining, device=device
    )

    eval_pipeline = eval_dino_features(
        dino_model=feature_model,
        n_last_blocks=config_dict['use_nblocks'],
        dataloader=dataloaders['train'],
        str_labels=class_names
    )
    fig = eval_pipeline.run_evaluation()
    fig.savefig('tsne_visualization.png')

if __name__ == "__main__":
    main()