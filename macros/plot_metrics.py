import json
import numpy as np
import matplotlib.pyplot as plt
import re

input_dir = '/nfs/data/1/nitish/dino_output/cvn_properrun/'
out_dir = input_dir+'plots/'

f_train_metrics = input_dir + 'training_metrics.json'
f_grads = input_dir + 'grads/grad_history_10epoch0009.json'

with open(f_train_metrics, 'r') as f:
    metrics = None
    for i, line in enumerate(f):
        metrics_dict = json.loads(line)
        dtype = [(key, type(metrics_dict[key])) for key in metrics_dict.keys()]
        # print(dtype)
        if i==0:
            metrics = np.array([tuple(metrics_dict.values())], dtype=dtype)
        else:
            metrics = np.append(metrics, np.array([tuple(metrics_dict.values())], dtype=dtype), axis=0)
aux = ['masked_patches_ratio']
losses_keys = [k for k in metrics.dtype.names if 'loss' in k or k in aux]

fig, axes = plt.subplots(3, 2, figsize=(5*2, 4*3))

for i, k in enumerate(losses_keys):
    if i < 6:  # Only plot first 6 losses (3x2 = 6 subplots)
        row = i // 2  # Calculate row index
        col = i % 2   # Calculate column index
        axes[row, col].plot(metrics['iteration'], metrics[k], label=k)
        axes[row, col].set_xlabel('Iteration', fontsize=15)
        axes[row, col].set_ylabel('loss', fontsize=15)
        axes[row, col].grid(True)
        axes[row, col].legend(fontsize=15)

# Hide empty subplots if you have fewer than 6 losses
for i in range(len(losses_keys), 6):
    row = i // 2
    col = i % 2
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(out_dir+'metrics.pdf')

def plot_grad(epoch=9):
    f_grads = input_dir + 'grads/grad_history_10epoch000%d.json' % epoch
    grad_metrics = {}
    with open(f_grads, 'r') as f:
        try:
            grads = json.load(f)
        except json.decoder.JSONDecodeError:
            return
        for key in grads.keys():
            grad_metrics[key] = np.array(grads[key]['norms'])


    fig, axes = plt.subplots(5, 4, figsize=(5*4, 4*5))
    for i, key in enumerate(grad_metrics.keys()):
        row = i // 4
        col = i % 4
        indices = np.arange(len(grad_metrics[key]))
        axes[row, col].plot(indices, grad_metrics[key], label=re.sub(r'_fsdp_wrapped_module', '', key))
        axes[row, col].set_xlabel('Iteration', fontsize=15)
        axes[row, col].set_ylabel('Gradient (L2-Norm)', fontsize=15)
        axes[row, col].grid(True)
        axes[row, col].legend(fontsize=15)

    plt.tight_layout()
    plt.savefig(out_dir+'grads_epoch%d.pdf'%epoch)

for i in range(10):
    plot_grad(i)
