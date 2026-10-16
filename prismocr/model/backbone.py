"""Multi-scale anisotropic backbone retained from the v3 design."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class AnisotropicRepDW(nn.Module):
    """Depthwise 3x3, 1x7, 3x1 and identity branches with explicit fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.dw_3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn_3x3 = nn.BatchNorm2d(channels)
        self.dw_1x7 = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=False)
        self.bn_1x7 = nn.BatchNorm2d(channels)
        self.dw_3x1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels, bias=False)
        self.bn_3x1 = nn.BatchNorm2d(channels)
        self.bn_identity = nn.BatchNorm2d(channels)
        self.fused_conv: Optional[nn.Conv2d] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_conv is not None:
            return self.fused_conv(x)
        return (
            self.bn_3x3(self.dw_3x3(x))
            + self.bn_1x7(self.dw_1x7(x))
            + self.bn_3x1(self.dw_3x1(x))
            + self.bn_identity(x)
        )

    @staticmethod
    def _fuse_bn(weight: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        weight = weight.float()
        scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
        scale = scale.reshape(-1, 1, 1, 1)
        bias = bn.bias.float() - bn.running_mean.float() * scale.reshape(-1)
        return weight * scale, bias

    @torch.no_grad()
    def fuse_for_inference(self) -> None:
        if self.fused_conv is not None:
            return
        if self.training:
            raise RuntimeError("call eval() before fusing anisotropic branches")
        device = self.dw_3x3.weight.device
        dtype = self.dw_3x3.weight.dtype
        kernel = torch.zeros((self.channels, 1, 3, 7), device=device, dtype=torch.float32)
        bias = torch.zeros(self.channels, device=device, dtype=torch.float32)

        w, b = self._fuse_bn(self.dw_3x3.weight, self.bn_3x3)
        kernel[:, :, :, 2:5] += w
        bias += b
        w, b = self._fuse_bn(self.dw_1x7.weight, self.bn_1x7)
        kernel[:, :, 1:2, :] += w
        bias += b
        w, b = self._fuse_bn(self.dw_3x1.weight, self.bn_3x1)
        kernel[:, :, :, 3:4] += w
        bias += b
        identity = torch.zeros_like(kernel)
        identity[:, 0, 1, 3] = 1.0
        w, b = self._fuse_bn(identity, self.bn_identity)
        kernel += w
        bias += b

        fused = nn.Conv2d(self.channels, self.channels, (3, 7), padding=(1, 3), groups=self.channels, bias=True)
        fused = fused.to(device=device, dtype=dtype)
        fused.weight.copy_(kernel.to(dtype))
        fused.bias.copy_(bias.to(dtype))
        self.fused_conv = fused
        del self.dw_3x3, self.bn_3x3, self.dw_1x7, self.bn_1x7
        del self.dw_3x1, self.bn_3x1, self.bn_identity


class RepBlock(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.5) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.dw = AnisotropicRepDW(channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.pw2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Start the residual branch at identity to avoid unstable gradients on
        # blank pages while the anisotropic branch weights are still random.
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn2(self.pw2(self.act(self.bn1(self.pw1(self.dw(x))))))

    def fuse_for_inference(self) -> None:
        self.dw.fuse_for_inference()


@contextmanager
def _batch_norm_recompute_context(module: nn.Module) -> Iterator[None]:
    """Run the same BN graph but roll back running-stat updates after recomputation."""
    batch_norms = [layer for layer in module.modules() if isinstance(layer, nn.modules.batchnorm._BatchNorm)]
    previous = [
        (
            layer,
            layer.running_mean.clone() if layer.running_mean is not None else None,
            layer.running_var.clone() if layer.running_var is not None else None,
            layer.num_batches_tracked.clone() if layer.num_batches_tracked is not None else None,
        )
        for layer in batch_norms
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for layer, running_mean, running_var, num_batches_tracked in previous:
                if running_mean is not None:
                    layer.running_mean.copy_(running_mean)
                if running_var is not None:
                    layer.running_var.copy_(running_var)
                if num_batches_tracked is not None:
                    layer.num_batches_tracked.copy_(num_batches_tracked)


class PrismBackbone(nn.Module):
    """Return stride-4/8/16 features for glyph and structure streams."""

    def __init__(self, dim: int = 160) -> None:
        super().__init__()
        self.dim = dim
        self.activation_checkpointing = False
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.stage4 = nn.Sequential(RepBlock(64, 2.0), RepBlock(64, 2.0))
        self.down8 = nn.Sequential(
            nn.Conv2d(64, 112, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(112),
            nn.SiLU(inplace=True),
        )
        self.stage8 = nn.Sequential(*(RepBlock(112, 2.5) for _ in range(3)))
        self.down16 = nn.Sequential(
            nn.Conv2d(112, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.stage16 = nn.Sequential(*(RepBlock(dim, 2.5) for _ in range(3)))
        self.cpe16 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def _run_stage(self, stage: nn.Module, value: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.activation_checkpointing):
            return stage(value)
        return checkpoint(
            stage,
            value,
            use_reentrant=False,
            context_fn=lambda: (nullcontext(), _batch_norm_recompute_context(stage)),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if images.shape[-2] % 16 or images.shape[-1] % 16:
            raise ValueError("image height and width must be multiples of backbone stride 16")
        if torch.is_autocast_enabled(images.device.type):
            # Keep the input stem block in FP32 under autocast. On flat/blank
            # pages the first BN can amplify gradients enough that casting its
            # input gradient back to FP16 overflows before GradScaler sees it.
            with torch.autocast(device_type=images.device.type, enabled=False):
                stem = self.stem[2](self.stem[1](self.stem[0](images.to(self.stem[0].weight.dtype))))
            stem = self.stem[3:](stem)
        else:
            stem = self.stem(images)
        f4 = self._run_stage(self.stage4, stem)
        f8 = self._run_stage(self.stage8, self.down8(f4))
        f16 = self._run_stage(self.stage16, self.down16(f8))
        return f4, f8, f16 + self.cpe16(f16)

    def fuse_for_inference(self) -> None:
        if self.training:
            raise RuntimeError("call eval() before fusing the backbone")
        for module in list(self.modules()):
            if isinstance(module, RepBlock):
                module.fuse_for_inference()
