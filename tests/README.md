# Note:
- The script visualize_attention.py doesn't work with XFormers yet.
- The model used for testing is a backbone model from the dinov2 github repository. No obvious patterns of what the model learned during its training was observed.
- This script is similar to visualize_attention.py from dinov1 but adapted to dinov2. The script used for dinov1 models works fine.

# How to visualize Attention

Sources: 
- https://github.com/facebookresearch/dinov2
- https://github.com/facebookresearch/dino


In the first version of dino, the authors did in fact visualize attention maps and made
the code available. So basically, we just change the code a little bit from the second version, 
such that it fits the first one.

The goal is to get the attentions of the transformer (for each head) from the last layer. 
Then, we can use those attention-scores to draw the heatmaps.

In the Dinov2 repo we have the vision transformer implemented in `dinov2/models/vision_transformer.py`.
Here we can add a function to the `DinoVisionTransformer` class, that extracts the last selfattention.
Consequently, we can merge the `forward_features` 
method from dinov2 and the `get_last_selfattention` method from dinov1:

```python
def get_last_self_attention(self, x, masks=None):
    if isinstance(x, list):
        return self.forward_features_list(x, masks)
        
    x = self.prepare_tokens_with_masks(x, masks)
    
    # Run through model, at the last block just return the attention.
    for i, blk in enumerate(self.blocks):
        if i < len(self.blocks) - 1:
            x = blk(x)
        else: 
            return blk(x, return_attention=True)
```

Furthermore, we have to change the `forward` function of the 
`Block` (dinov2/layers/block.py) class, such that we can hand 
it the `return_attention` argument:

```python
    def forward(self, x: Tensor, return_attention=False) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        
        # Add this 2 lines
        if return_attention:
            return self.attn(self.norm1(x), return_attn=True)
            
        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x
```

As this is only the base class, and `NestedTensorBlock` is the actual used class inheriting from
`Block`, we need to also change the foward function (in the same file) here:

```python
def forward(self, x_or_x_list, return_attention=False):
        if isinstance(x_or_x_list, Tensor):
            # Change the following line
            # return super().forward(x_or_x_list)
            return super().forward(x_or_x_list, return_attention)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
```

Note that now it only works if you feed in a single Tensor, for a list and using XFORMERS, you basically need
to do the same for the `forward_nested` function.

Finally, we need to change the `forward` method of the `Attention` class (dinov2/layers/attention.py), such that 
we can hand it the `return_attn` attribute:

```python
def forward(self, x: Tensor, return_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Add those 2 lines
        if return_attn:
            return attn
        return x
```

Again, the `Attention` class is only the base class, so you have to also adjust the `MemEffAttention` class:

```python
class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            # Change this line
            # return super().forward(x)
            return super().forward(x, return_attn)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

Note that this now will only work when you don't use XFORMERS (nested tensors).

Finally, we can use the `visualize_attention.py` file from dinov1 and change it a
little bit to visualize the heatmaps! Keep in mind that you have to change the line
img = Image.open('cow-on-the-beach2.jpg') to your specific filename and you need
to download the weights from the dinov2 github repo, obviously. 

Additionally, note the weird line `attentions[:, 10] = 0`. I found out that the attention
scores are very high for one specific pixel, over all attention heads. This is why i 
print the maximum pixel_idx, and manually *deactivate* it for every individual image. 
I don't really understand why this is happening...







