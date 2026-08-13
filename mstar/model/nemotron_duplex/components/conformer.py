"""Fast-Conformer STT perception stack (``stt_model.perception.*``).

NeMo Fast-Conformer: mel preprocessor -> dw-striding conv subsampling ->
24 Conformer layers (Macaron FF, rel-pos MHA, conv module) -> projection into
the nano LLM hidden space (1024 -> 4480).

Ported to exact checkpoint parity. FF/attention/conv linears are bias-free;
norms are LayerNorm; the conv module's BatchNorm keeps only weight+bias in the
checkpoint (running stats are absent — a Phase-4 numerics note). Forward is
Phase 4 (audio-in); the tree here is load-verified.
"""
from __future__ import annotations

from torch import nn

from mstar.model.nemotron_duplex.components._util import param

D = 1024          # encoder d_model
N_HEADS = 8
HEAD_DIM = 128
FF = 4096
NUM_LAYERS = 24
SUB_CH = 256      # subsampling channels
LLM_HIDDEN = 4480


class _Featurizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fb = param(1, 128, 257)      # mel filterbank
        self.window = param(400)          # STFT window


class Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.featurizer = _Featurizer()


class ConvSubsampling(nn.Module):
    """dw-striding subsampling: Conv2d, ReLU, (depthwise, pointwise, ReLU)x2 -> Linear."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, SUB_CH, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(SUB_CH, SUB_CH, kernel_size=3, stride=2, padding=1, groups=SUB_CH),
            nn.Conv2d(SUB_CH, SUB_CH, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(SUB_CH, SUB_CH, kernel_size=3, stride=2, padding=1, groups=SUB_CH),
            nn.Conv2d(SUB_CH, SUB_CH, kernel_size=1),
        )
        self.out = nn.Linear(4352, D)  # 256 * ceil(freq/8) -> d_model


class _RelPosMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(D, D, bias=False)
        self.linear_k = nn.Linear(D, D, bias=False)
        self.linear_v = nn.Linear(D, D, bias=False)
        self.linear_out = nn.Linear(D, D, bias=False)
        self.linear_pos = nn.Linear(D, D, bias=False)
        self.pos_bias_u = param(N_HEADS, HEAD_DIM)
        self.pos_bias_v = param(N_HEADS, HEAD_DIM)


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(D, FF, bias=False)
        self.linear2 = nn.Linear(FF, D, bias=False)


class _ConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(D, 2 * D, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv1d(D, D, kernel_size=9, padding=4, groups=D, bias=False)
        self.batch_norm = nn.BatchNorm1d(D)
        self.pointwise_conv2 = nn.Conv1d(D, D, kernel_size=1, bias=False)


class ConformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(D)
        self.feed_forward1 = _FeedForward()
        self.norm_self_att = nn.LayerNorm(D)
        self.self_attn = _RelPosMHA()
        self.norm_conv = nn.LayerNorm(D)
        self.conv = _ConvModule()
        self.norm_feed_forward2 = nn.LayerNorm(D)
        self.feed_forward2 = _FeedForward()
        self.norm_out = nn.LayerNorm(D)


class ConformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_encode = ConvSubsampling()
        self.layers = nn.ModuleList([ConformerLayer() for _ in range(NUM_LAYERS)])


class Perception(nn.Module):
    """Full perception stack: preprocessor + conformer encoder + LLM projection."""

    def __init__(self):
        super().__init__()
        self.preprocessor = Preprocessor()
        self.encoder = ConformerEncoder()
        self.proj = nn.Linear(D, LLM_HIDDEN)  # 1024 -> 4480, into nano hidden

    @staticmethod
    def remap(name: str) -> str | None:
        prefix = "stt_model.perception."
        return name[len(prefix):] if name.startswith(prefix) else None

    def load_weights(self, weights):
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)
