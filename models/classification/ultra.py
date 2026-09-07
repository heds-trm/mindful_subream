import torch
import torch.nn as nn
from monai.utils import ensure_tuple_rep
from monai.networks.layers import trunc_normal_
from monai.networks.blocks.transformerblock import TransformerBlock
import numpy as np
import warnings
from typing import Sequence

from mindful_core.models.representation.encoders.factory import ENCODERS_REGISTER


class UlTra(nn.Module):
    def __init__(
            self,
            in_channels: int,
            image_size: Sequence[int] | int,
            phase_count: int,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            output_dimension: int = None,
            use_class_token: bool = True,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            qkv_bias: bool = False,
            pooling: str | int = "first",
            norm_outputs: bool = False,
    ) -> None:
        super(UlTra, self).__init__()

        # region Check arguments
        if not (0 <= dropout_rate <= 1):
            raise ValueError("`dropout_rate` should be between 0 and 1.")

        if (hidden_size % num_heads) != 0:
            raise ValueError("`hidden_size` should be divisible by num_heads.")

        if in_channels != 1:
            raise NotImplementedError("`in_channels` currently must be 1.")
        # endregion

        self.in_channels = in_channels
        self.image_size = ensure_tuple_rep(image_size, dim=spatial_dims)
        self.phase_count = phase_count
        self.output_dimension = output_dimension
        self.spatial_dims = spatial_dims

        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.use_class_token = use_class_token
        self.dropout_rate = dropout_rate
        self.pooling = pooling
        self.norm_outputs = norm_outputs

        self.phase_embedding = UlTraPhaseEmbeddingBlock(in_channels=in_channels,
                                                        image_size=image_size,
                                                        phase_count=phase_count,
                                                        hidden_size=hidden_size,
                                                        num_heads=num_heads,
                                                        dropout_rate=dropout_rate,
                                                        spatial_dims=spatial_dims)

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) \
            if use_class_token else None

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size=hidden_size,
                             mlp_dim=mlp_dim,
                             num_heads=num_heads,
                             dropout_rate=dropout_rate,
                             qkv_bias=qkv_bias)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size) \
            if norm_outputs else None

        self.projector = nn.Linear(hidden_size, output_dimension) \
            if self.project_outputs else None
## rajout de output intermediates
    def forward(self, data: torch.Tensor, output_intermediates: bool = False, *args, **kwargs) -> torch.Tensor:
        data = self.phase_embedding(data)

        if self.use_class_token:
            data = self.insert_class_token(data)

        for block in self.blocks:
            data = block(data)

        data = self.apply_pooling(data)

        if self.norm_outputs:
            data = self.norm(data)

        if self.project_outputs:
            data = self.projector(data)

        return data

    def insert_class_token(self, data: torch.Tensor) -> torch.Tensor:
        class_token = self.class_token.expand(data.size(0), -1, -1)
        return torch.cat([class_token, data], dim=1)

    def apply_pooling(self, inputs: torch.Tensor) -> torch.Tensor:
        if (self.pooling is None) or (self.pooling == "none"):
            return inputs

        elif self.pooling in ["first", "cls", 0, 1]:
            return inputs[:, 0]

        elif self.pooling == "mean":
            return inputs.mean(dim=1)

        elif self.pooling == "max":
            return inputs.max(dim=1).values # j'ai rajouté ça car erreur avec le pooling max (13.04.2026)

        elif self.pooling in ["mul", "prod"]:
            return inputs.prod(dim=1)

        elif isinstance(self.pooling, int):
            return inputs[:, :self.pooling]

        raise NotImplementedError("Unknown pooling method `{}`".format(self.pooling))

    @property
    def project_outputs(self) -> bool:
        return ((self.output_dimension is not None) and
                (self.output_dimension != self.hidden_size))


class UlTraPhaseEmbeddingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            image_size: Sequence[int] | int,
            phase_count: int,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:
        super(UlTraPhaseEmbeddingBlock, self).__init__()

        # region Check arguments
        if not (0 <= dropout_rate <= 1):
            raise ValueError("`dropout_rate` should be between 0 and 1.")

        if (hidden_size % num_heads) != 0:
            raise ValueError("`hidden_size` should be divisible by num_heads.")
        # endregion

        self.in_channels = in_channels
        self.image_size = ensure_tuple_rep(image_size, dim=spatial_dims)
        self.phase_count = phase_count
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.spatial_dims = spatial_dims
        self.patch_dim = int(in_channels * np.prod(image_size))

        if self.patch_dim > 1024:
            warnings.warn("Patch size seems very high ({}), " \
                          "are you sure you meant the image " \
                          "to be this big ?".format(self.patch_dim))

        self.patch_embeddings = nn.Linear(self.patch_dim, self.hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.phase_count, self.hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    # region Init weights
    def _init_weights(self) -> None:
        self._init_trunc_normal(self.position_embeddings)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            self._init_trunc_normal(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, val=0.0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @staticmethod
    def _init_trunc_normal(weights: nn.Parameter) -> None:
        trunc_normal_(weights, mean=0.0, std=0.02, a=-2.0, b=2.0)

    # endregion

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size = data.size(0)
        if self.in_channels != 1:
            data = data.transpose(1, 2)
        data = data.reshape(batch_size, -1, self.patch_dim)
        patch_count = data.size(1)

        data = self.patch_embeddings(data)
        data += self.position_embeddings[:, :patch_count]
        data = self.dropout(data)
        return data

def make_ultra_encoder(image_size: int | tuple[int, int, int],
                       in_channels: int,
                       phase_count: int,
                       num_layers: int,
                       num_heads: int,
                       output_dimension: int,
                       spatial_dims: int,
                       hidden_size: int = None,
                       mlp_dim: int = None,
                       **kwargs):
    hidden_size = output_dimension if hidden_size is None else hidden_size
    mlp_dim = output_dimension * 2 if mlp_dim is None else mlp_dim
    encoder = UlTra(in_channels=in_channels, image_size=image_size, phase_count=phase_count,
                    hidden_size=hidden_size, mlp_dim=mlp_dim, output_dimension=output_dimension,
                    num_layers=num_layers, num_heads=num_heads,
                    spatial_dims=spatial_dims, **kwargs)
    return encoder

ENCODERS_REGISTER["ultra"] = make_ultra_encoder
