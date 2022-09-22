import torch
from torch import nn
from operator import itemgetter

from mmcv.runner import BaseModule
from ..builder import MODELS

# Helper functions
def exists(val):
    return val is not None


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape
def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(
            1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None, fc_layer=True):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.fc_layer = fc_layer
        if self.fc_layer:
            self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        if self.fc_layer:
            out = self.to_out(out)
        return out


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            for f in block:
                x = x + f(x)
        return x


@MODELS.register_module()
class AxialAttentionTransformer(BaseModule):

    def __init__(self,
                 dim=384,
                 num_dimensions=3,
                 depth=1,
                 heads=8,
                 dim_heads=None,
                 dim_index=2,
                 axial_pos_emb_shape=(3, 128, 128),
                 fc_layer_attn=False,
                 init_cfg=None
                 ):
        super(AxialAttentionTransformer, self).__init__(init_cfg=init_cfg)
        # self.model = AxialTempTransformer(dim=dim,
        #                                   num_dimensions=num_dimensions,
        #                                   depth=depth,
        #                                   heads=heads,
        #                                   dim_index=dim_index,
        #                                   axial_pos_emb_shape=axial_pos_emb_shape,
        #                                   fc_layer_attn=fc_layer_attn)
        
        permutations = calculate_permutations(num_dimensions, dim_index)
        permutations = [permutations[1],permutations[2], permutations[0]] # Spatial attention first
        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(axial_pos_emb_shape) else nn.Identity()
        
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList([PermuteToFrom(permutation, 
                PreNorm(dim, SelfAttention(dim, heads, dim_heads, fc_layer=fc_layer_attn))) for permutation in permutations])
            layers.append(attn_functions)
            to_out = nn.ModuleList([PermuteToFrom(permutations[-1], PreNorm(dim, nn.Linear(dim, dim)))])
            layers.append(to_out)

        self.layers = Sequential(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)
