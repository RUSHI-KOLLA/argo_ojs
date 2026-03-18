"""
src/models/architectures.py
----------------------------
Teacher (YOLOv8l) and Student (YOLOv8s) model wrappers for
Knowledge Distillation training.

- FrozenTeacher: loads pretrained YOLOv8l weights, freezes all params,
  and registers forward hooks at backbone layers 4, 6, 9 for
  intermediate feature extraction.

- Student: builds YOLOv8s from YAML with a custom detection head
  adapted to the project's class count, plus matching hooks.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

HOOK_LAYERS: List[int] = [4, 6, 9]


# ----------------------------------------------------------------
# Frozen Teacher (YOLOv8l)
# ----------------------------------------------------------------

class FrozenTeacher(nn.Module):
    """
    Wraps a pretrained YOLOv8l model for use as a Knowledge Distillation teacher.

    All parameters are frozen (no gradients). Forward hooks on backbone
    layers 4, 6, and 9 capture intermediate feature maps used for
    feature-level distillation loss.

    Usage:
        teacher = FrozenTeacher('path/to/best_teacher.pt').to(device)
        with torch.no_grad():
            out = teacher(imgs)
        feats = teacher.features()  # {'l4': tensor, 'l6': tensor, 'l9': tensor}
    """

    def __init__(self, weights_path: str):
        super().__init__()
        self._feats: Dict[str, torch.Tensor] = {}

        # Load YOLO, extract underlying nn.Module (avoid keeping YOLO as child)
        y = YOLO(str(weights_path))
        self.model = y.model
        del y

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self._register_hooks()

        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Teacher loaded: {total / 1e6:.1f}M params (frozen)")

    def _register_hooks(self) -> None:
        def hook_fn(name: str):
            def fn(module, inp, out):
                feat = out[0] if isinstance(out, tuple) else out
                self._feats[name] = feat.detach()
            return fn

        for idx in HOOK_LAYERS:
            self.model.model[idx].register_forward_hook(hook_fn(f"l{idx}"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._feats.clear()
        with torch.no_grad():
            return self.model(x)

    def features(self) -> Dict[str, torch.Tensor]:
        """Return the most recent intermediate feature maps."""
        return dict(self._feats)


# ----------------------------------------------------------------
# Student (YOLOv8s)
# ----------------------------------------------------------------

class Student(nn.Module):
    """
    YOLOv8s student model built from YAML config with a custom
    detection head adapted to the project's number of classes.

    Forward hooks on layers 4, 6, 9 mirror the teacher's hooks
    to enable feature-level distillation.

    Usage:
        student = Student(nc=19).to(device)
        out = student(imgs)
        feats = student.features()
    """

    def __init__(self, nc: int = 19):
        super().__init__()
        self._feats: Dict[str, torch.Tensor] = {}

        # Build YOLOv8s from YAML
        self.model = DetectionModel(cfg="yolov8s.yaml", ch=3, nc=nc)
        self._adapt_detect_head(nc)
        self.model.nc = nc
        self.model.names = {i: str(i) for i in range(nc)}
        self._register_hooks()

        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Student built: {total / 1e6:.1f}M params | nc={nc}")

    def _adapt_detect_head(self, nc: int) -> None:
        """Replace the final classification conv in each detection scale."""
        det = self.model.model[-1]
        det.nc = nc
        for b in det.cv3:
            old = b[-1]
            new_conv = nn.Conv2d(
                old.in_channels, nc,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=True
            )
            nn.init.normal_(new_conv.weight, 0, 0.01)
            nn.init.constant_(new_conv.bias, -4.5)
            b[-1] = new_conv
        det.no = det.nc + det.reg_max * 4

    def _register_hooks(self) -> None:
        def hook_fn(name: str):
            def fn(module, inp, out):
                feat = out[0] if isinstance(out, tuple) else out
                self._feats[name] = feat
            return fn

        for idx in HOOK_LAYERS:
            self.model.model[idx].register_forward_hook(hook_fn(f"l{idx}"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._feats.clear()
        return self.model(x)

    def features(self) -> Dict[str, torch.Tensor]:
        """Return the most recent intermediate feature maps."""
        return dict(self._feats)


# ----------------------------------------------------------------
# Feature Adapters (Student → Teacher channel alignment)
# ----------------------------------------------------------------

def build_feature_adapters(
    student: Student,
    teacher: FrozenTeacher,
    device: torch.device
) -> nn.ModuleDict:
    """
    Build 1×1 conv adapters to align student feature channels to teacher channels.
    One adapter per hook layer (l4, l6, l9).

    Channels are inferred by running a dummy forward pass.

    Args:
        student: Initialized Student model.
        teacher: Initialized FrozenTeacher model.
        device: Target device.

    Returns:
        nn.ModuleDict of adapters keyed by 'l4', 'l6', 'l9'.
    """
    dummy = torch.zeros(1, 3, 640, 640, device=device)

    with torch.no_grad():
        teacher(dummy)
        student(dummy)

    t_feats = teacher.features()
    s_feats = student.features()

    adapters = {}
    for key in [f"l{i}" for i in HOOK_LAYERS]:
        if key in s_feats and key in t_feats:
            in_ch  = s_feats[key].shape[1]
            out_ch = t_feats[key].shape[1]
            adapters[key] = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            print(f"  Adapter {key}: {in_ch} → {out_ch} channels")

    return nn.ModuleDict(adapters).to(device)
