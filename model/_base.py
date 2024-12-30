import torch
from torch import nn
from typing import Type

class CreatePatchesLayer(nn.Module):
  def __init__(
    self,
    patch_size: int,
    strides: int,
  ) -> None:
    """Init Variables."""
    super().__init__()
    self.unfold_layer = nn.Unfold(
      kernel_size=patch_size, stride=strides
    )

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    """Forward Pass to Create Patches."""
    patched_images = self.unfold_layer(images)
    return patched_images.permute((0, 2, 1))
  

class PatchEmbeddingLayer(nn.Module):
  def __init__(self, num_patches: int, batch_size: int, patch_size: int, embed_dim: int, device: torch.device) -> None:
    """Init Function."""
    
    super().__init__()
    self.num_patches = num_patches
    self.patch_size = patch_size
    self.batch_size = batch_size
    
    self.position_emb = nn.Embedding(
      num_embeddings=num_patches, embedding_dim=embed_dim
    )
    self.projection_layer = nn.Linear(
      patch_size * patch_size * 3, embed_dim
    )
    self.class_parameter = nn.Parameter(
      torch.rand(batch_size, 1, embed_dim).to(device),
      requires_grad=True,
    )
    self.device = device

  def forward(self, patches: torch.Tensor) -> torch.Tensor:
    """Forward Pass."""
    positions = (
      torch.arange(start=0, end=self.num_patches + 1, step=1)
      .to(self.device)
      .unsqueeze(dim=0)
    )
    
    patches = self.projection_layer(patches)
    class_tokens = self.class_parameter.expand(self.batch_size, -1, -1)
    encoded_patches = torch.cat(
      (class_tokens, patches), dim=1
    ) + self.position_emb(positions)
    return encoded_patches
  

def create_mlp_block(
    input_features: int,
    output_features: list[int],
    activation_function: Type[nn.Module],
    dropout_rate: float,
) -> nn.Module:
    """Create a Feed Forward Network for the Transformer Layer."""
    layer_list = []
    for idx in range(  # pylint: disable=consider-using-enumerate
        len(output_features)
    ):
        if idx == 0:
            linear_layer = nn.Linear(
                in_features=input_features, out_features=output_features[idx]
            )
        else:
            linear_layer = nn.Linear(
                in_features=output_features[idx - 1],
                out_features=output_features[idx],
            )
        dropout = nn.Dropout(p=dropout_rate)
        layers = nn.Sequential(
            linear_layer, activation_function(), dropout
        )
        layer_list.append(layers)
    return nn.Sequential(*layer_list)


class TransformerBlock(nn.Module):
  """Transformer Block Layer."""

  def __init__(self, num_heads: int, key_dim: int, embed_dim: int, ff_dim: int, dropout_rate: float = 0.1,) -> None:
    """Init variables and layers."""
    super().__init__()
    self.layer_norm_input = nn.LayerNorm(
      normalized_shape=embed_dim, eps=1e-6
    )
    self.attn = nn.MultiheadAttention(
      embed_dim=embed_dim,
      num_heads=num_heads,
      kdim=key_dim,
      vdim=key_dim,
      batch_first=True,
    )

    self.dropout_1 = nn.Dropout(p=dropout_rate)
    self.layer_norm_1 = nn.LayerNorm(
      normalized_shape=embed_dim, eps=1e-6
    )
    self.layer_norm_2 = nn.LayerNorm(
      normalized_shape=embed_dim, eps=1e-6
    )
    self.ffn = create_mlp_block(
      input_features=embed_dim,
      output_features=[ff_dim, embed_dim],
      activation_function=nn.GELU,
      dropout_rate=dropout_rate,
    )


  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Forward Pass."""
    layer_norm_inputs = self.layer_norm_input(inputs)
    attention_output, _ = self.attn(
      query=layer_norm_inputs,
      key=layer_norm_inputs,
      value=layer_norm_inputs,
    )
    attention_output = self.dropout_1(attention_output)
    out1 = self.layer_norm_1(inputs + attention_output)
    ffn_output = self.ffn(out1)
    output = self.layer_norm_2(out1 + ffn_output)
    return output
  

class ViTBlock(nn.Module):
  """ViT Model for Image Classification."""

  def __init__(self, patch_size, n_trans, device: torch.device, **kwargs) -> None:
    """Init Function."""
    super().__init__()
    self.create_patch_layer = CreatePatchesLayer(patch_size, patch_size)
    self.patch_embedding_layer = PatchEmbeddingLayer(
      kwargs['num_patches'], kwargs['batch_size'], patch_size, kwargs['projection_dim'], device
    )
    self.transformer_layers = nn.ModuleList()

    for _ in range(n_trans):
      self.transformer_layers.append(
        TransformerBlock(
          kwargs['num_heads'], kwargs['projection_dim'], kwargs['projection_dim'], kwargs['feed_forward_dim']
        )
      )

    self.mlp_block = create_mlp_block(
      input_features=kwargs['projection_dim'],
      output_features=kwargs['mlp_head_units'],
      activation_function=nn.GELU,
        dropout_rate=0.5,
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward Pass."""
    x = self.create_patch_layer(x)
    x = self.patch_embedding_layer(x)
    
    for transformer_layer in self.transformer_layers:
      x = transformer_layer(x)
    


    return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CNNBlock, self).__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # nn.GroupNorm(8, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cnn_block(x)
        return x


class MultiCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, n_conv):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(n_conv):
            self.layers.append(CNNBlock(in_channels, out_channels, padding=padding))
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class OutputBlock(nn.Module):
    def __init__(self, in_channels, num_class):
        super().__init__()

        self.output_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, num_class, kernel_size=1),
        )
    
    def forward(self, x):
        return self.output_block(x)
