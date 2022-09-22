import torch
from torch import nn

from mmcv.runner import BaseModule
from ..builder import MODELS
from ..model_utils import ConvLSTM, AxialTempTransformer


@MODELS.register_module()
class AxialAttentionTransformer(BaseModule):

    def __init__(self,
                dim = 384,
                num_dimensions = 3,
                depth = 1,
                heads = 8,
                dim_index = 2,
                axial_pos_emb_shape = (3, 128, 128), # 0.2
                fc_layer_attn=False,
                init_cfg=None
    ):
        super(AxialAttentionTransformer, self).__init__(init_cfg=init_cfg)
        self.model = AxialTempTransformer(dim=dim,
                num_dimensions=num_dimensions,
                depth=depth,
                heads=heads,
                dim_index=dim_index,
                axial_pos_emb_shape=axial_pos_emb_shape,
                fc_layer_attn=fc_layer_attn)

        
    def forward(self, x):
        x = self.model.forward(x)
        return x

