# Model using Multi-Resolution HuBERT pretrained model + AASIST architecture
# Adapted from https://github.com/TakHemlata/SSL_Anti-spoofing/blob/a16b0684b02f4f7b8de9778a5727eb0c67f23281/model.py (MIT License) and
# https://github.com/s3prl/s3prl/blob/70637b0b851b308caf0f17d6ccd1980ce6c7c596/s3prl/upstream/multires_hubert/hubert_model.py (Apache 2.0 license)

import os
import math
import json
import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, is_dataclass
from enum import Enum, EnumMeta
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForAudioXVector
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import XVectorOutput


logger = logging.getLogger(__name__)

def merge_with_parent(dc: dataclass, cfg: dict):
    assert is_dataclass(dc)
    assert type(cfg) is dict
    cfg = deepcopy(cfg)

    def fix_cfg(cfg):
        target_keys = set(dc.__dataclass_fields__.keys())
        for k in list(cfg.keys()):
            if k not in target_keys:
                del cfg[k]

    fix_cfg(cfg)
    assert len(cfg) > 0
    return dc(**cfg)


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


def relu_squared(x: torch.Tensor):
    return torch.nn.functional.relu(x).pow(2)


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    def gelu_accurate(x):
        if not hasattr(gelu_accurate, "_a"):
            gelu_accurate._a = math.sqrt(2 / math.pi)
        return (
            0.5
            * x
            * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
        )

    def gelu(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x.float()).type_as(x)

    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "gelu_fast",  # deprecated
        "gelu_accurate",
        "tanh",
        "linear",
    ]


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = torch.nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)
    return mod


def index_put(tensor, indices, value):
    tensor[indices] = value
    return tensor


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2
    return torch.nn.functional.pad(x, (*pad_offset, 0, remainder), value=value), remainder


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(int(mask_other), mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )

        mask[i, mask_idc] = True

    return mask


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
        key: str,
        value: Dict[str, Optional[torch.Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


class FairseqDropout(torch.nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return torch.nn.functional.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


@with_incremental_state
class MultiheadAttention(torch.nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        # TODO: pass in config rather than string.
        # config defined in xformers.components.attention.AttentionConfig
        xformers_att_config: Optional[str] = None,
        xformers_blocksparse_layout: Optional[
            torch.Tensor
        ] = None,  # This should be part of the config
        xformers_blocksparse_blocksize: Optional[
            int
        ] = 16,  # This should be part of the config
    ):
        super().__init__()

        def eval_str_dict(x, type=dict):
            if x is None:
                return None
            if isinstance(x, str):
                x = eval(x)
            return x

        xformers_att_config = eval_str_dict(xformers_att_config)
        self.use_xformers = xformers_att_config is not None
        assert not self.use_xformers, "Do not use xformers in S3PRL"

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            torch.nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            torch.nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            torch.nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            torch.nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = torch.nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = torch.nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.beam_size = 1
        self.reset_parameters()

        self.onnx_trace = False
        self.skip_embed_dim_check = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            torch.nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            torch.nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            torch.nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            torch.nn.init.xavier_uniform_(self.k_proj.weight)
            torch.nn.init.xavier_uniform_(self.v_proj.weight)
            torch.nn.init.xavier_uniform_(self.q_proj.weight)

        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            torch.nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            torch.nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            torch.nn.init.xavier_normal_(self.bias_v)

    def _get_reserve_head_index(self, num_heads_to_keep: int):
        k_proj_heads_norm = []
        q_proj_heads_norm = []
        v_proj_heads_norm = []

        for i in range(self.num_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        self.k_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(self.k_proj.bias[start_idx:end_idx])).tolist()
            )
            q_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        self.q_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(self.q_proj.bias[start_idx:end_idx])).tolist()
            )
            v_proj_heads_norm.append(
                torch.sum(
                    torch.abs(
                        self.v_proj.weight[
                            start_idx:end_idx,
                        ]
                    )
                ).tolist()
                + torch.sum(torch.abs(self.v_proj.bias[start_idx:end_idx])).tolist()
            )

        heads_norm = []
        for i in range(self.num_heads):
            heads_norm.append(
                k_proj_heads_norm[i] + q_proj_heads_norm[i] + v_proj_heads_norm[i]
            )

        sorted_head_index = sorted(
            range(self.num_heads), key=lambda k: heads_norm[k], reverse=True
        )
        reserve_head_index = []
        for i in range(num_heads_to_keep):
            start = sorted_head_index[i] * self.head_dim
            end = (sorted_head_index[i] + 1) * self.head_dim
            reserve_head_index.append((start, end))
        return reserve_head_index

    def _adaptive_prune_heads(self, reserve_head_index: List[Tuple[int, int]]):
        new_q_weight = []
        new_q_bias = []
        new_k_weight = []
        new_k_bias = []
        new_v_weight = []
        new_v_bias = []
        new_out_proj_weight = []

        for ele in reserve_head_index:
            start_idx, end_idx = ele
            new_q_weight.append(
                self.q_proj.weight[
                    start_idx:end_idx,
                ]
            )
            new_q_bias.append(self.q_proj.bias[start_idx:end_idx])

            new_k_weight.append(
                self.k_proj.weight[
                    start_idx:end_idx,
                ]
            )

            new_k_bias.append(self.k_proj.bias[start_idx:end_idx])

            new_v_weight.append(
                self.v_proj.weight[
                    start_idx:end_idx,
                ]
            )
            new_v_bias.append(self.v_proj.bias[start_idx:end_idx])

            new_out_proj_weight.append(self.out_proj.weight[:, start_idx:end_idx])

        new_q_weight = torch.cat(new_q_weight).detach()
        new_k_weight = torch.cat(new_k_weight).detach()
        new_v_weight = torch.cat(new_v_weight).detach()
        new_out_proj_weight = torch.cat(new_out_proj_weight, dim=-1).detach()
        new_q_weight.requires_grad = True
        new_k_weight.requires_grad = True
        new_v_weight.requires_grad = True
        new_out_proj_weight.requires_grad = True

        new_q_bias = torch.cat(new_q_bias).detach()
        new_q_bias.requires_grad = True

        new_k_bias = torch.cat(new_k_bias).detach()
        new_k_bias.requires_grad = True

        new_v_bias = torch.cat(new_v_bias).detach()
        new_v_bias.requires_grad = True

        self.q_proj.weight = torch.nn.Parameter(new_q_weight)
        self.q_proj.bias = torch.nn.Parameter(new_q_bias)

        self.k_proj.weight = torch.nn.Parameter(new_k_weight)
        self.k_proj.bias = torch.nn.Parameter(new_k_bias)

        self.v_proj.weight = torch.nn.Parameter(new_v_weight)
        self.v_proj.bias = torch.nn.Parameter(new_v_bias)

        self.out_proj.weight = torch.nn.Parameter(new_out_proj_weight)

        self.num_heads = len(reserve_head_index)
        self.embed_dim = self.head_dim * self.num_heads
        self.q_proj.out_features = self.embed_dim
        self.k_proj.out_features = self.embed_dim
        self.v_proj.out_features = self.embed_dim

    def _set_skip_embed_dim_check(self):
        self.skip_embed_dim_check = True

    def _pad_masks(
        self,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if attn_mask is not None:
            shape = attn_mask.size()[:-1] + torch.Size([1])
            attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(shape)], dim=-1)
        if key_padding_mask is not None:
            shape = key_padding_mask.size()[:-1] + torch.Size([1])
            key_padding_mask = torch.cat(
                [
                    key_padding_mask,
                    key_padding_mask.new_zeros(shape),
                ],
                dim=-1,
            )
        return key_padding_mask, attn_mask

    def _add_bias(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        bsz: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self.bias_k is not None
        assert self.bias_v is not None
        k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        key_padding_mask, attn_mask = self._pad_masks(
            key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        return k, v, key_padding_mask, attn_mask

    def _append_zero_attn(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        zero_attn_shape = k.size()[:-2] + torch.Size([1]) + k.size()[-1:]
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=-2
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=-2
        )
        key_padding_mask, attn_mask = self._pad_masks(
            key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        return k, v, key_padding_mask, attn_mask

    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
            # The Multihead attention implemented in pytorch forces strong dimension check
            # for input embedding dimention and K,Q,V projection dimension.
            # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
            # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
            and not self.skip_embed_dim_check
        ):
            assert key is not None and value is not None

            if self.use_xformers:
                return self._xformers_attn_forward(
                    query, key, value, key_padding_mask, need_weights, attn_mask
                )

            else:
                return torch.nn.functional.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout_module.p,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training or self.dropout_module.apply_during_inference,
                    key_padding_mask,
                    need_weights,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[
                        :, :, 0, :
                    ]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.view(
                            -1, self.beam_size, key_padding_mask.size(1)
                        )[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k, v, attn_mask, key_padding_mask, bsz
            )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = (
                k.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                kv_bsz = _prev_key.size(0)
                prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                assert kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(
                    kv_bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[torch.Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=kv_bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(
                kv_bsz, self.num_heads, -1, self.head_dim
            )
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn_weights = torch.einsum(
                "bxhtd,bhsd->bxhts",
                q.view((kv_bsz, -1, self.num_heads) + q.size()[1:]),
                k.view((kv_bsz, self.num_heads) + k.size()[1:]),
            )
            attn_weights = attn_weights.reshape((-1,) + attn_weights.size()[-2:])
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.view(
                    kv_bsz, -1, self.num_heads, tgt_len, src_len
                )
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        def softmax_supporting_onnx_trace(x, dim: int, onnx_trace: bool = False):
            if onnx_trace:
                return torch.nn.functional.softmax(x.float(), dim=dim)
            else:
                return torch.nn.functional.softmax(x, dim=dim, dtype=torch.float32)

        attn_weights_float = softmax_supporting_onnx_trace(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = torch.einsum(
                "bxhts,bhsd->bxhtd",
                attn_probs.view(
                    (
                        kv_bsz,
                        -1,
                        self.num_heads,
                    )
                    + attn_probs.size()[1:]
                ),
                v.view(
                    (
                        kv_bsz,
                        self.num_heads,
                    )
                    + v.size()[1:]
                ),
            )
            attn = attn.reshape((-1,) + attn.size()[-2:])
        else:
            attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[torch.Tensor],
        prev_key_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[torch.Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        new_order: torch.Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention:
                        if input_buffer_k.size(0) * self.beam_size == new_order.size(0):
                            return incremental_state
                        elif self.beam_size > 1:
                            input_buffer[k] = input_buffer_k.index_select(
                                0,
                                new_order.reshape(-1, self.beam_size)[:, 0]
                                // self.beam_size,
                            )
                        else:
                            input_buffer[k] = input_buffer_k.index_select(0, new_order)
                    else:
                        input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def set_beam_size(self, beam_size):
        """Used for effiecient beamable enc-dec attention"""
        self.beam_size = beam_size

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[torch.Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        buffer: Dict[str, Optional[torch.Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(self, max_len, d_model):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x : Input tensor T X B X C.
        Returns:
            torch.Tensor: Encoded tensor T X B X C.

        """
        x = x.transpose(0, 1)  # Change TBC to BTC
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        pos_emb = pos_emb.transpose(0, 1)  # change to TBC
        return pos_emb


class GumbelVectorQuantizer(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=torch.nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = torch.nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        torch.nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = torch.nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                torch.nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = torch.nn.Linear(self.input_dim, groups * num_vars)
            torch.nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            torch.nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars**self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
            .index_select(0, indices)
            .view(self.num_vars**self.groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
            n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars**exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):
        result = {"num_vars": self.num_vars * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = torch.nn.functional.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        result["x"] = x

        return result


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SamePad(torch.nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransposeLast(torch.nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = torch.nn.functional.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32GroupNorm(torch.nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = torch.nn.functional.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class ConvFeatureExtractionModel(torch.nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = torch.nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                torch.nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return torch.nn.Sequential(
                    make_conv(),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    torch.nn.GELU(),
                )
            elif is_group_norm:
                return torch.nn.Sequential(
                    make_conv(),
                    torch.nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    torch.nn.GELU(),
                )
            else:
                return torch.nn.Sequential(make_conv(), torch.nn.Dropout(p=dropout), torch.nn.GELU())

        in_d = 1
        self.conv_layers = torch.nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl
            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


def make_conv_pos(e, k, g):
    pos_conv = torch.nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    torch.nn.init.normal_(pos_conv.weight, mean=0, std=std)
    torch.nn.init.constant_(pos_conv.bias, 0)

    pos_conv = torch.nn.utils.parametrizations.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = torch.nn.Sequential(pos_conv, SamePad(k), torch.nn.GELU())

    return pos_conv

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])
LAYER_TYPE_CHOICES = ChoiceEnum(["transformer", "conformer"])


@dataclass
class Wav2Vec2Config:
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )
    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"},
    )
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help": "dimensionality increase for inner quantizer layers (if depth > 1)"
        },
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_before: bool = False
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={
            "help": "crop convolutional feature extractor output such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})


class TransformerSentenceEncoderLayer(torch.nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(self.activation_dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = torch.nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = torch.nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)


class TransformerEncoder(torch.nn.Module):
    def build_encoder_layer(self, args: Wav2Vec2Config):
        if args.layer_type == "transformer":
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
        elif args.layer_type == "conformer":
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args.encoder_ffn_embed_dim,
                attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                activation_fn="swish",
                attn_type=args.attn_type,
                use_fp16=args.fp16,
                pos_enc_type="abs",
            )
        return layer

    def __init__(
        self,
        args: Wav2Vec2Config,
        skip_pos_conv: bool = False,
        override_encoder_layer: int = None,
    ):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return torch.nn.Sequential(
                    *[
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            torch.nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim, k, args.conv_pos_groups, num_layers
            )

        elif skip_pos_conv:
            self.pos_conv = None
        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
            )

        if override_encoder_layer is None:
            encoder_layers = args.encoder_layers
        else:
            encoder_layers = override_encoder_layer

        self.layers = torch.nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(encoder_layers)]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
    ):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        if self.pos_conv is not None:
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else b,
                    c[:-pad_length],
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


@dataclass
class MultiresHubertPretrainingConfig:
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    #     label_rate: 1,2,2,5
    #                 (imply (1,2), (2,5))
    #     if base label_rate = 50
    #     (1,2), (2,5) --> label rates 50, 25, 10
    label_rate_ratios: list = field(
        default_factory=lambda: [1, 2],
        metadata={"help": "tuple for label rates e.g., [(1,2), (2,5)]"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys " "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )


@dataclass
class MultiresHubertConfig:
    label_rate: float
    #     label_rate: 1,2,2,5
    #                 (imply (1,2), (2,5))
    #     if base label_rate = 50
    #     (1,2), (2,5) --> label rates 50, 25, 10
    label_rate_ratios: List[int] = field(
        default_factory=lambda: [1, 2],
        metadata={"help": "list of label rates e.g., [1,2, 2,5]"},
    )

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    # the blocks for each label rate
    encoder_layers: int = field(
        default="2",
        metadata={
            "help": "num encoder layers in the each block (one sub module of the U-net)"
        },
    )
    override_encoder_layers: str = field(
        default="",
        metadata={
            "help": "specific layer numbers for each block (one sub module of the U-net) for the training"
        },
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )
    conv_adapator_kernal: int = field(
        default=7, metadata={"help": "kernal size for conv adaptor"}
    )
    use_plain_updownsample: bool = field(
        default=False, metadata={"help": "whether to use plain up downsample"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=True,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )
    use_single_target: bool = field(
        default=False,
        metadata={
            "help": "whether to use single data (in that case, we will compute the fixed label rate)"
        },
    )
    use_single_prediction: bool = field(
        default=False,
        metadata={
            "help": "if true, we will not conduct mlm prediction in low resolution in the middle"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})


class MultiresHubertModel(torch.nn.Module):
    def __init__(
        self,
        cfg: MultiresHubertConfig,
        task_cfg: MultiresHubertPretrainingConfig,
        dictionaries: List[Any],
    ) -> None:
        super().__init__()

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            torch.nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        # Estimate label rates
        assert (
            cfg.label_rate_ratios != "None"
        ), "without ratios, the model is exactly as the Hubert model"
        self.label_rate_ratios = []
        self.base_rate = cfg.label_rate
        self.label_rates = []
        self.downsample_modules = torch.nn.ModuleList()
        self.upsample_modules = torch.nn.ModuleList()
        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        self.use_single_target = cfg.use_single_target
        self.use_single_prediction = cfg.use_single_prediction
        self.use_plain_updownsample = cfg.use_plain_updownsample

        # For decide the override encoder layers, so that the layer number is not equally distributed
        if cfg.override_encoder_layers != "":
            self.override_encoder_layers = eval(cfg.override_encoder_layers)
            assert (
                len(self.override_encoder_layers) % 2 == 1
            ), "must be odd number of layers if specify detailed layers"
            assert (
                len(self.override_encoder_layers) // 2
                == len(cfg.label_rate_ratios) // 2
            ), "number of override encoder layers must match the label rate ratios information"
            self.len_encoder_modules = len(self.override_encoder_layers)
        else:
            self.override_encoder_layers = None
            self.len_encoder_modules = None

        # use different layers instead of equally distributed ones
        middle_override_encoder_layer = (
            self.override_encoder_layers[self.len_encoder_modules // 2]
            if self.override_encoder_layers is not None
            else None
        )
        skip_middle_pos_conv = False if len(cfg.label_rate_ratios) < 2 else True

        self.middle_encoder = TransformerEncoder(
            cfg,
            skip_pos_conv=skip_middle_pos_conv,
            override_encoder_layer=middle_override_encoder_layer,
        )

        first_pos_conv = False  # only enable pos_conv for the first encoder
        raw_label_rate_ratios = cfg.label_rate_ratios
        for i in range(len(raw_label_rate_ratios) // 2):
            # check if have override encoder layers
            if self.override_encoder_layers is not None:
                override_encoder_layer = self.override_encoder_layers[i]
                override_decoder_layer = self.override_encoder_layers[
                    self.len_encoder_modules - 1 - i
                ]
            else:
                override_encoder_layer, override_decoder_layer = None, None

            self.label_rate_ratios.append(
                (raw_label_rate_ratios[i * 2], raw_label_rate_ratios[i * 2 + 1])
            )

            if self.use_plain_updownsample:
                self.downsample_modules.append(
                    ConvDownsampler(
                        k=cfg.conv_adapator_kernal,
                        label_rate=(
                            (
                                raw_label_rate_ratios[i * 2],
                                raw_label_rate_ratios[i * 2 + 1],
                            )
                        ),
                        dropout=0.0,
                        channels=cfg.encoder_embed_dim,
                        activation=torch.nn.GELU(),
                        log_compression=False,
                        skip_connections=True,
                        highway=True,
                        residual_scale=0.4,
                    )
                )
            else:
                self.downsample_modules.append(
                    ConvAdapter(
                        k=cfg.conv_adapator_kernal,
                        label_rate=(
                            (
                                raw_label_rate_ratios[i * 2],
                                raw_label_rate_ratios[i * 2 + 1],
                            )
                        ),
                        dropout=0.0,
                        channels=cfg.encoder_embed_dim,
                        activation=torch.nn.GELU(),
                        log_compression=False,
                        skip_connections=True,
                        highway=True,
                        residual_scale=0.4,
                    )
                )
            if not first_pos_conv:
                self.encoders.append(
                    TransformerEncoder(
                        cfg, override_encoder_layer=override_encoder_layer
                    )
                )  # TODO(jiatong): add conformer options
                first_pos_conv = True
            else:
                self.encoders.append(
                    TransformerEncoder(
                        cfg,
                        skip_pos_conv=True,
                        override_encoder_layer=override_encoder_layer,
                    )
                )
            if self.use_plain_updownsample:
                self.upsample_modules.append(
                    ConvUpsampler(
                        k=cfg.conv_adapator_kernal,
                        label_rate=(
                            (
                                raw_label_rate_ratios[i * 2 + 1],
                                raw_label_rate_ratios[i * 2],
                            )
                        ),
                        dropout=0.0,
                        channels=cfg.encoder_embed_dim,
                        activation=torch.nn.GELU(),
                        log_compression=False,
                        skip_connections=True,
                        highway=True,
                        residual_scale=0.4,
                    )
                )
            else:
                self.upsample_modules.append(
                    ConvAdapter(
                        k=cfg.conv_adapator_kernal,
                        label_rate=(
                            (
                                raw_label_rate_ratios[i * 2 + 1],
                                raw_label_rate_ratios[i * 2],
                            )
                        ),
                        dropout=0.0,
                        channels=cfg.encoder_embed_dim,
                        activation=torch.nn.GELU(),
                        log_compression=False,
                        skip_connections=True,
                        highway=True,
                        residual_scale=0.4,
                    )
                )
            self.decoders.append(
                TransformerEncoder(
                    cfg,
                    skip_pos_conv=True,
                    override_encoder_layer=override_decoder_layer,
                )
            )

        base_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feature_ds_rates = [base_ds_rate]
        running_rate = self.base_rate

        if cfg.use_single_target:
            self.label_rates = self.base_rate
        else:
            self.label_rates.append(self.base_rate)

        for label_rate_ratio in self.label_rate_ratios:
            upsample_rate, downsample_rate = label_rate_ratio
            if (base_ds_rate * upsample_rate) % downsample_rate != 0:
                print(
                    "base rate: {} cannot be ideally processed with downsample rate {}".format(
                        base_ds_rate, downsample_rate
                    )
                )

            base_ds_rate = base_ds_rate * downsample_rate // upsample_rate
            self.feature_ds_rates.append(base_ds_rate)

            if not cfg.use_single_target:
                running_rate = running_rate * upsample_rate // downsample_rate
                self.label_rates.append(running_rate)
        self.label_nums = len(
            self.feature_ds_rates
        )  # the number of labels for prediction (activate at iter 2)

        if type(self.label_rates) == float:
            self.feat2tar_ratios = [
                self.feature_ds_rates[i] * self.label_rates / task_cfg.sample_rate
                for i in range(len(self.feature_ds_rates))
            ]
        else:
            self.feat2tar_ratios = [
                self.feature_ds_rates[i] * self.label_rates[i] / task_cfg.sample_rate
                for i in range(len(self.feature_ds_rates))
            ]

        # self.feat2tar_ratios = self.feat2tar_ratios[::-1]

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = torch.nn.Dropout(cfg.dropout_input)
        self.dropout_features = torch.nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = 0.0
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = torch.nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.layer_norm = LayerNorm(self.embed)

        self.predictor_head_num = 1 if self.use_single_prediction else self.label_nums

        self.target_glu = None
        if cfg.target_glu:
            self.target_glus = torch.nn.ModuleList()
            for i in range(self.predictor_head_num):
                self.target_glus.append(
                    torch.nn.Sequential(torch.nn.Linear(final_dim, final_dim * 2), torch.nn.GLU())
                )

        self.untie_final_proj = cfg.untie_final_proj
        self.final_projs = torch.nn.ModuleList()

        # Note(jiatong): we do not have untie cases for multires hubert
        for i in range(self.predictor_head_num):
            self.final_projs.append(torch.nn.Linear(cfg.encoder_embed_dim, final_dim))

        # modules below are not needed during fine-tuning
        self.multires_classes = []
        self.label_embs_concat = torch.nn.ParameterList()

        for i in range(self.predictor_head_num):
            # if dictionaries[i] is None:
            #     self.label_embs_concat.append(None)
            #     continue
            # TODO(jiatong): for skipping case
            if self.use_single_target:
                num_classes = len(dictionaries[0])
            else:
                num_classes = len(dictionaries[i])
            self.multires_classes.append(num_classes)
            self.label_embs_concat.append(
                torch.nn.Parameter(torch.FloatTensor(num_classes, final_dim))
            )
            torch.nn.init.uniform_(self.label_embs_concat[i])

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: MultiresHubertConfig, task):
        """Build a new model instance."""

        model = MultiresHubertModel(cfg, task.cfg, task.dictionaries)
        return model

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        feat2tar_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels

        feat_tsz = features.size(1)

        # skip if no target is provided
        if target is None:
            return features, None, None
        targ_tsz = target.size(1)
        if feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / feat2tar_ratio)
            features = features[:, :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * feat2tar_ratio
        target = target[:, target_inds.long()]
        return features, target

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.forward_features(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        def align_size_sum(feat1, pad1, feat2):
            common_size = min(feat1.size(1), feat2.size(1))

            return (
                feat1[:, :common_size] + feat2[:, :common_size],
                pad1[:, :common_size],
            )

        # process encoders
        res_outputs = []  # final output for different resolution
        multi_mask_indices = []  # mask indices for different resolution
        residuals = []  # record the x in encoders
        padding_masks = []  # final padding masks
        # The encoder has (self.label_nums - 1) blocks
        for i in range(self.label_nums - 1):
            x, _ = self.encoders[i](x, padding_mask=padding_mask, layer=None)
            residuals.append(x)
            x, padding_mask, mask_indices = self.downsample_modules[i](
                x, padding=padding_mask, mask_indices=mask_indices
            )

        residual = self.middle_encoder(x, padding_mask=padding_mask, layer=None)[0]
        x = x + residual
        res_outputs.append(x)

        # process decoders
        # The encoder has (self.label_nums - 1) blocks
        padding_masks.append(padding_mask)
        multi_mask_indices.append(mask_indices)
        residuals.reverse()  # NOTE(jiatong): reverse res_output to match corresponding input
        for i in range(self.label_nums - 1):
            x, padding_mask, mask_indices = self.upsample_modules[i](
                x, padding=padding_mask, mask_indices=mask_indices
            )
            x, _ = self.decoders[i](x, padding_mask=padding_mask, layer=None)
            x, padding_mask = align_size_sum(x, padding_mask, residuals[i])
            res_outputs.append(x)
            padding_masks.append(padding_mask)
            multi_mask_indices.append(mask_indices)

        # NOTE(jiatong): need reverse of taßßßrget list to allow matched target-representation
        res_outputs.reverse()
        padding_masks.reverse()
        multi_mask_indices.reverse()
        if target_list is not None:
            new_target_list = []
            for i in range(self.label_nums):
                if self.use_single_target:
                    res_outputs[i], reformat_target_list = self.forward_targets(
                        res_outputs[i], target_list[0], self.feat2tar_ratios[i]
                    )
                    new_target_list.append(reformat_target_list)
                else:
                    if target_list[i] is not None:
                        res_outputs[i], reformat_target_list = self.forward_targets(
                            res_outputs[i], target_list[i], self.feat2tar_ratios[i]
                        )
                        new_target_list.append(reformat_target_list)
                    else:
                        # Append a None target list then it won't be used to calculate loss
                        new_target_list.append(None)
                if padding_masks[i] is not None:
                    padding_masks[i] = self.forward_padding_mask(
                        res_outputs[i], padding_masks[i]
                    )
                if multi_mask_indices[i] is not None:
                    multi_mask_indices[i] = self.forward_padding_mask(
                        res_outputs[i], multi_mask_indices[i]
                    )

        if features_only:
            # NOTE(jiatong): need to reverse back
            res_outputs.reverse()
            return {
                "x": res_outputs,
                "padding_mask": padding_mask,
                "features": features,
            }

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        multires_record = {
            "logit_m_list": [],
            "logit_u_list": [],
            "padding_mask": [],
            "features_pen": [],
        }

        logit_m_list, logit_u_list = [], []
        for j in range(self.label_nums):
            if new_target_list[j] is None:
                continue  # skip empty targets
            label_embs_list = self.label_embs_concat[j].split(
                [self.multires_classes[j]], 0
            )
            # set the variables (after the set, the procedure is the same as hubert)
            # all the elements are list with only one element (to simulate the normal hubert process)
            x = res_outputs[j]
            target = new_target_list[j]
            padding_mask = padding_masks[j]
            mask_indices = multi_mask_indices[j]
            final_proj = self.final_projs[j]

            if not self.skip_masked:
                masked_indices = torch.logical_and(~padding_mask, mask_indices)
                proj_x_m = final_proj(x[masked_indices])
                logit_m_list.append(
                    compute_pred(proj_x_m, target[masked_indices], label_embs_list[0])
                )
            else:
                logit_m_list.append(None)

            if not self.skip_nomask:
                nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
                proj_x_u = final_proj(x[nomask_indices])
                logit_u_list.append(
                    compute_pred(proj_x_u, target[nomask_indices], label_embs_list[0])
                )
            else:
                logit_u_list.append(None)

            multires_record["logit_m_list"].append(logit_m_list)
            multires_record["logit_u_list"].append(logit_u_list)
            multires_record["padding_mask"].append(padding_mask)
            multires_record["features_pen"].append(features_pen)

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        last_layer: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        if last_layer:
            feature = feature[-1]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None


class ConvAdapter(torch.nn.Module):
    """Conv adapter that combines two modules with different label rate with downsample or upsample.
    To allow different ratios than integer, two convs are utilized with first to upsample (numerator)
    and the second to downsample (denominator)"""

    def __init__(
        self,
        k,
        label_rate,
        dropout,
        channels,
        activation,
        log_compression=False,
        skip_connections=True,
        highway=True,
        residual_scale=0.4,
        non_affine_group_norm=False,
    ):
        super().__init__()

        def downsample_block(channel, k, stride):
            return torch.nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                torch.nn.Conv1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=(k - 1) // 2,
                ),
                torch.nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        def upsample_block(channel, k, stride):
            return torch.nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                torch.nn.ConvTranspose1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=0,  # padding=(k - 1) // 2,
                    output_padding=(stride - 1),
                ),
                torch.nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        assert len(label_rate) == 2, "label_rate should be sized two to apply fusion"
        # Lout =(Lin~H~R1)~Wstride~H~R2~Wpadding+dilation~W(kernel_size~H~R1)+output_padding+1
        self.upsample_conv = upsample_block(channels, k, label_rate[0])
        self.downsample_conv = downsample_block(channels, k, label_rate[1])

        self.upsample_rate, self.downsample_rate = label_rate
        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.highway = highway
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x, padding=None, mask_indices=None):
        # Assume x1 = (B, T, C) as input
        x = x.permute(0, 2, 1)
        residual_before_upsample = x
        x = self.upsample_conv(x)
        upsample_size = x.size(2)

        # conduct upsample
        if self.skip_connections:
            residual_upsample = torch.repeat_interleave(
                residual_before_upsample, self.upsample_rate, dim=2
            )
            upsample_size = min(upsample_size, residual_upsample.size(2))
            x = (
                x[..., :upsample_size] + residual_upsample[..., :upsample_size]
            ) * self.residual_scale

        residual_before_downsample = x
        x = self.downsample_conv(x)
        downsample_size = x.size(2)

        if self.skip_connections:
            residual_downsample = residual_before_downsample[
                ..., :: self.downsample_rate
            ]
            downsample_size = min(x.size(2), residual_downsample.size(2))
            x = (
                x[..., :downsample_size] + residual_downsample[..., :downsample_size]
            ) * self.residual_scale

        if self.highway:
            residual_after_sample = residual_upsample[..., :: self.downsample_rate]
            final_size = min(x.size(2), residual_after_sample.size(2))
            x = (
                x[..., :final_size] + residual_after_sample[..., :final_size]
            ) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        x = x.permute(0, 2, 1)

        # process padding
        if padding is not None:
            padding = torch.repeat_interleave(padding, self.upsample_rate, dim=1)
            padding = padding[..., :: self.downsample_rate]
            padding = padding[..., : x.size(1)]

        # process mask indices
        if mask_indices is not None:
            mask_indices = torch.repeat_interleave(
                mask_indices, self.upsample_rate, dim=1
            )
            mask_indices = mask_indices[..., :: self.downsample_rate]
            mask_indices = mask_indices[..., : x.size(1)]
        return x, padding, mask_indices


class ConvDownsampler(torch.nn.Module):
    """Conv downsampler that combines two modules with different label rate with downsample or upsample.
    To allow different ratios than integer, two convs are utilized with first to upsample (numerator)
    and the second to downsample (denominator)"""

    def __init__(
        self,
        k,
        label_rate,
        dropout,
        channels,
        activation,
        log_compression=False,
        skip_connections=True,
        highway=True,
        residual_scale=0.4,
        non_affine_group_norm=False,
    ):
        super().__init__()

        def downsample_block(channel, k, stride):
            return torch.torch.nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                torch.nn.Conv1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=(k - 1) // 2,
                ),
                torch.nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        assert len(label_rate) == 2, "label_rate should be sized two to apply fusion"
        self.downsample_conv = downsample_block(channels, k, label_rate[1])

        upsample_rate, self.downsample_rate = label_rate
        assert upsample_rate == 1, "must be 1 to perform downsample only"
        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.highway = highway  # Useless as placeholder
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x, padding=None, mask_indices=None):
        # Assume x1 = (B, T, C) as input
        x = x.permute(0, 2, 1)

        residual_before_downsample = x
        x = self.downsample_conv(x)
        downsample_size = x.size(2)

        if self.skip_connections:
            residual_downsample = residual_before_downsample[
                ..., :: self.downsample_rate
            ]
            downsample_size = min(x.size(2), residual_downsample.size(2))
            x = (
                x[..., :downsample_size] + residual_downsample[..., :downsample_size]
            ) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        x = x.permute(0, 2, 1)

        # process padding
        if padding is not None:
            padding = padding[..., :: self.downsample_rate]
            padding = padding[..., : x.size(1)]

        # process mask indices
        if mask_indices is not None:
            mask_indices = mask_indices[..., :: self.downsample_rate]
            mask_indices = mask_indices[..., : x.size(1)]
        return x, padding, mask_indices


class ConvUpsampler(torch.nn.Module):
    """Conv upsampler that combines two modules with different label rate with downsample or upsample.
    To allow different ratios than integer, two convs are utilized with first to upsample (numerator)
    and the second to downsample (denominator)"""

    def __init__(
        self,
        k,
        label_rate,
        dropout,
        channels,
        activation,
        log_compression=False,
        skip_connections=True,
        highway=True,
        residual_scale=0.4,
        non_affine_group_norm=False,
    ):
        super().__init__()

        def upsample_block(channel, k, stride):
            return torch.nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                torch.nn.ConvTranspose1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=0,  # padding=(k - 1) // 2,
                    output_padding=(stride - 1),
                ),
                torch.nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        assert len(label_rate) == 2, "label_rate should be sized two to apply fusion"
        # Lout =(Lin~H~R1)~Wstride~H~R2~Wpadding+dilation~W(kernel_size~H~R1)+output_padding+1
        self.upsample_conv = upsample_block(channels, k, label_rate[0])

        self.upsample_rate, downsample_rate = label_rate
        assert downsample_rate == 1, "must be 1 to perform downsample only"
        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.highway = highway  # Useless
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x, padding=None, mask_indices=None):
        # Assume x1 = (B, T, C) as input
        x = x.permute(0, 2, 1)
        residual_before_upsample = x
        x = self.upsample_conv(x)
        upsample_size = x.size(2)

        # conduct upsample
        if self.skip_connections:
            residual_upsample = torch.repeat_interleave(
                residual_before_upsample, self.upsample_rate, dim=2
            )
            upsample_size = min(upsample_size, residual_upsample.size(2))
            x = (
                x[..., :upsample_size] + residual_upsample[..., :upsample_size]
            ) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        x = x.permute(0, 2, 1)

        # process padding
        if padding is not None:
            padding = torch.repeat_interleave(padding, self.upsample_rate, dim=1)
            padding = padding[..., : x.size(1)]

        # process mask indices
        if mask_indices is not None:
            mask_indices = torch.repeat_interleave(
                mask_indices, self.upsample_rate, dim=1
            )
            mask_indices = mask_indices[..., : x.size(1)]
        return x, padding, mask_indices

#---------AASIST back-end------------------------#
# Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu and Nicholas Evans.
# AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks.
# In Proc. ICASSP 2022, pp: 6367--6371.'''


class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = torch.nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = torch.nn.Linear(in_dim, out_dim)
        self.proj_without_att = torch.nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = torch.nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = torch.nn.Dropout(p=0.2)

        # activate
        self.act = torch.nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = torch.nn.functional.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = torch.nn.Parameter(torch.FloatTensor(*size))
        torch.nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = torch.nn.Linear(in_dim, in_dim)
        self.proj_type2 = torch.nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = torch.nn.Linear(in_dim, out_dim)
        self.att_projM = torch.nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = torch.nn.Linear(in_dim, out_dim)
        self.proj_without_att = torch.nn.Linear(in_dim, out_dim)

        self.proj_with_attM = torch.nn.Linear(in_dim, out_dim)
        self.proj_without_attM = torch.nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = torch.nn.BatchNorm1d(num_features=out_dim)

        # dropout for inputs
        self.input_drop = torch.nn.Dropout(p=0.2)

        # activate
        self.act = torch.nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        # directional edge for master node
        master = self._update_master(x, master)
        # projection
        x = self._project(x, att_map)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = torch.nn.functional.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # apply temperature
        att_map = att_map / self.temp

        att_map = torch.nn.functional.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = torch.nn.Parameter(torch.FloatTensor(*size))
        torch.nn.init.xavier_normal_(out)
        return out


class GraphPool(torch.nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = torch.nn.Sigmoid()
        self.proj = torch.nn.Linear(in_dim, 1)
        self.drop = torch.nn.Dropout(p=p) if p > 0 else torch.nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class Residual_block(torch.nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = torch.nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = torch.nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = torch.nn.SELU(inplace=True)
        self.bn2 = torch.nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = torch.nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = torch.nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )
        else:
            self.downsample = False
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out


class MRHubertAASISTConfig(PretrainedConfig):
    model_type = "MRHubertAASIST"


class MRHubertAASIST(PreTrainedModel):
    config_class = MRHubertAASISTConfig
    main_input_name = "input_values"
    base_model_prefix = "mrhubert"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)

        task_cfg = {}
        model_cfg = {}
        dictionaries_symbols = [[]]
        s3prl_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mrhubert.json")
        with open(s3prl_config_path, 'r', encoding='utf-8') as config_file:
            cfgs = json.load(config_file)
            task_cfg = cfgs["task_cfg"]
            model_cfg = cfgs["model_cfg"]
            dictionaries_symbols = cfgs["dictionaries_symbols"]
        task_cfg = merge_with_parent(MultiresHubertPretrainingConfig, task_cfg)
        model_cfg = merge_with_parent(MultiresHubertConfig, model_cfg)
        self.mrhubert = MultiresHubertModel(model_cfg, task_cfg, dictionaries_symbols)

        aasist_config = dict({
            'first_conv': 128,
            'in_channels': 1,
            'filts': [128, [1, 32], [32, 32], [32, 64], [64, 64]], # no. of filter channel in residual blocks
            'gat_dims': [64, 32],
            'pool_ratios': [0.5, 0.5, 0.5, 0.5],
            'temperatures': [2.0, 2.0, 100.0, 100.0],
            'drop_way': 0.2,
            'drop_last': 0.5,
        })
        self.LL = torch.nn.Linear(config.hidden_size, aasist_config["first_conv"])

        self.first_bn = torch.nn.BatchNorm2d(num_features=1)
        self.first_bn1 = torch.nn.BatchNorm2d(num_features=64)
        self.drop_last = torch.nn.Dropout(aasist_config["drop_last"], inplace=True)
        self.drop_way = torch.nn.Dropout(aasist_config["drop_way"], inplace=True)
        self.selu = torch.nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Sequential(Residual_block(nb_filts=aasist_config["filts"][1], first=True)),
            torch.nn.Sequential(Residual_block(nb_filts=aasist_config["filts"][2])),
            torch.nn.Sequential(Residual_block(nb_filts=aasist_config["filts"][3])),
            torch.nn.Sequential(Residual_block(nb_filts=aasist_config["filts"][4])),
            torch.nn.Sequential(Residual_block(nb_filts=aasist_config["filts"][4])),
            torch.nn.Sequential(Residual_block(nb_filts=aasist_config["filts"][4])),
        )

        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(1,1)),
            torch.nn.SELU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 64, kernel_size=(1,1)),
        )
        # position encoding
        self.pos_S = torch.nn.Parameter(torch.randn(1, 42, aasist_config["filts"][-1][-1]))

        self.master1 = torch.nn.Parameter(torch.randn(1, 1, aasist_config["gat_dims"][0]))
        self.master2 = torch.nn.Parameter(torch.randn(1, 1, aasist_config["gat_dims"][0]))
        
        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(
            aasist_config["filts"][-1][-1],
            aasist_config["gat_dims"][0],
            temperature=aasist_config["temperatures"][0],
        )
        self.GAT_layer_T = GraphAttentionLayer(
            aasist_config["filts"][-1][-1],
            aasist_config["gat_dims"][0],
            temperature=aasist_config["temperatures"][1],
        )
        # HS-GAL layer 
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            aasist_config["gat_dims"][0],
            aasist_config["gat_dims"][1],
            temperature=aasist_config["temperatures"][2],
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            aasist_config["gat_dims"][1],
            aasist_config["gat_dims"][1],
            temperature=aasist_config["temperatures"][2],
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            aasist_config["gat_dims"][0],
            aasist_config["gat_dims"][1],
            temperature=aasist_config["temperatures"][2],
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            aasist_config["gat_dims"][1],
            aasist_config["gat_dims"][1],
            temperature=aasist_config["temperatures"][2],
        )

        # Graph pooling layers
        self.pool_S = GraphPool(aasist_config["pool_ratios"][0], aasist_config["gat_dims"][0], 0.3)
        self.pool_T = GraphPool(aasist_config["pool_ratios"][1], aasist_config["gat_dims"][0], 0.3)
        self.pool_hS1 = GraphPool(aasist_config["pool_ratios"][2], aasist_config["gat_dims"][1], 0.3)
        self.pool_hT1 = GraphPool(aasist_config["pool_ratios"][2], aasist_config["gat_dims"][1], 0.3)
        self.pool_hS2 = GraphPool(aasist_config["pool_ratios"][2], aasist_config["gat_dims"][1], 0.3)
        self.pool_hT2 = GraphPool(aasist_config["pool_ratios"][2], aasist_config["gat_dims"][1], 0.3)
       
        self.classifier = torch.nn.Linear(5 * aasist_config["gat_dims"][1], config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.mrhubert.parameters():
            param.requires_grad = False
        self.mrhubert.eval()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], XVectorOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.mrhubert.extract_features(
            input_values,
            padding_mask=attention_mask if attention_mask is not None else torch.zeros_like(input_values),
            mask=None,
        )[0][1]

        x = self.LL(hidden_states) #(bs,frame_number,feat_out_dim)

        # post-processing on front-end features
        x = x.transpose(1, 2)   #(bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1) # add channel 
        x = torch.nn.functional.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)

        # Self attention for spectral feature
        w1 = torch.nn.functional.softmax(w,dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S 
        
        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)
        
        # Self attention for temporal feature
        w2 = torch.nn.functional.softmax(w,dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)
       
        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        
        last_hidden = self.drop_last(last_hidden)
        logits = self.classifier(last_hidden)
        
        loss = None

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + last_hidden
            return ((loss,) + output) if loss is not None else output

        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=last_hidden,
            hidden_states=hidden_states,
            attentions=None,
        )

AutoConfig.register("MRHubertAASIST", MRHubertAASISTConfig)
AutoModelForAudioXVector.register(MRHubertAASISTConfig, MRHubertAASIST)
