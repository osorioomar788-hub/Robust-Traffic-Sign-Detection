# 🏛️ Phase 3 — Architecture Engineering: Final Report
## Robust Traffic Sign Detection | 25th MICAI
**Author:** Chief AI Architect  
**Phase Lead:** Amaury  
**Date:** 2026-03-06  
**Status:** ✅ COMPLETE — All validations passed

---

## 📋 Executive Summary

Phase 3 set out to solve two core weaknesses of the vanilla YOLOv8n architecture for the TT100K traffic sign detection domain:

| Problem | Root Cause | Our Solution |
|---------|-----------|--------------|
| **Missed small signs** (< 32×32 px in 2048×2048 images) | `nn.Upsample` uses naïve nearest-neighbor/bilinear interpolation — it fabricates spatial detail rather than learning it | **Sub-Pixel Convolution (ESPCN)** — a learnable upsampling module that recovers fine-grained spatial information through trained convolution filters |
| **Degraded performance in rain, fog, and night** | Standard feature fusion has no concept of *where* a sign is — attention is channel-only | **Coordinate Attention** — encodes both horizontal and vertical positional information, enabling the network to focus on sign-shaped regions even under degraded contrast |

### Result

We successfully modified the YOLOv8n neck by inserting **2 SubPixelConv** and **4 CoordAtt** modules—without modifying a single line of the Ultralytics library. The custom architecture:

- Builds from YAML ✓
- Passes a dry-run forward pass with input shape `(1, 3, 640, 640)` ✓
- Contains 3 detection heads at scales P3/P4/P5 ✓
- Recognizes all 143 TT100K classes ✓
- Totals ~4.8M parameters (vs. ~3.2M vanilla — a 50% increase that is well within real-time budgets)

---

## ⚙️ Design Patterns

### Pattern 1: Separation of Concerns

Phase 3 uses a strict three-file separation:

```
models/custom_modules.py    ← Pure PyTorch. Knows nothing about Ultralytics.
models/configs/yolov8_custom.yaml  ← Pure YAML. Declares architecture via names.
scripts/phase3_architecture_design.py  ← Glue code. Bridges the gap at runtime.
```

**Why this matters:** `custom_modules.py` can be unit-tested, linted, and reviewed in isolation. It imports only `torch` and `torch.nn`. The YAML is a declarative description with no executable logic. The glue script is the *only* file that touches Ultralytics internals, and it does so through a controlled, idempotent injection function.

### Pattern 2: Runtime Dependency Injection via Monkey-Patching

Ultralytics' `parse_model()` function resolves module names from the YAML by looking them up in its own namespace. It also maintains an internal `base_modules` frozenset that controls which modules receive automatic channel injection (the `c1`/`c2` wiring). Our classes are not in either of these.

Rather than forking or editing Ultralytics source files (which would break on every `pip upgrade`), we inject at runtime:

1. **Namespace injection** — `setattr(ultralytics.nn.modules, 'SubPixelConv', SubPixelConv)` makes `parse_model`'s `globals()[m]` resolution find our class.
2. **Source-level monkey-patch** — We read `parse_model`'s source via `inspect.getsource()`, insert our module names into the `base_modules` frozenset literal, and `exec()` the modified source back into the `tasks` module's global namespace.

This is aggressive but *contained*: the patch is idempotent (guarded by `_custom_patched`), reversible (just don't call it), and version-aware (it fails fast with a clear error if the source structure changes).

---

## 🔬 Deep Dive: `models/custom_modules.py`

### `SubPixelConv` — Learnable Upsampling

**Paper:** Shi et al., *"Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"*, CVPR 2016.

#### The Problem with `nn.Upsample`

YOLOv8's neck uses `nn.Upsample(scale_factor=2, mode='nearest')` to merge deep (semantically rich) features with shallow (spatially precise) features. But nearest-neighbor interpolation simply duplicates pixels — it adds *zero* new spatial information. For small traffic signs that occupy only a few pixels, this means the network is trying to classify objects from duplicated, blurry features.

#### The ESPCN Solution

Instead of upsampling *then* convolving, ESPCN convolves *then* rearranges:

```
Input (B, c1, H, W)
  │
  ├─ Conv2d(c1, c2 × r², kernel=3, pad=1, bias=False)
  │    └─ Expands channels to carry r² sub-pixel values per output channel
  │    └─ Shape: (B, c2·r², H, W)
  │
  ├─ nn.PixelShuffle(r)
  │    └─ Rearranges: Y[b, c, h·r+δh, w·r+δw] = X[b, c·r²+δh·r+δw, h, w]
  │    └─ Shape: (B, c2, H·r, W·r)
  │
  ├─ BatchNorm2d(c2)
  │    └─ Normalizes across output channels
  │
  └─ SiLU()
       └─ Activation (YOLOv8 convention)

Output (B, c2, H·r, W·r)
```

The key insight is that the `Conv2d` layer *learns* `r²` different upsampling filters per output channel. This is strictly more expressive than any fixed interpolation kernel.

#### Constructor Arguments

```python
def __init__(self, c1: int, c2: int, scale: int = 2):
```

| Arg | Source | Description |
|-----|--------|-------------|
| `c1` | Auto-injected by `parse_model` | Input channels from the previous layer's output |
| `c2` | YAML `args[0]` | Desired output channels (e.g., `512`, `256`) |
| `scale` | YAML `args[1]` or default `2` | Upscale factor — matches YOLOv8's 2× upsample stride |

**Why `c1` and `c2` are required:** Ultralytics' `parse_model()` calls every `base_module` as `Module(c1, c2, *args)`, where `c1` is automatically tracked from the previous layer's output channel count (after applying the width multiplier). If our constructor did not accept `c1` as the first positional argument, `parse_model` would crash with a `TypeError`.

---

### `CoordAtt` — Coordinate Attention

**Paper:** Hou et al., *"Coordinate Attention for Efficient Mobile Network Design"*, CVPR 2021.

#### Why Channel Attention Alone Isn't Enough

SE-Net (Squeeze-and-Excitation) style attention compresses the entire spatial grid into a single vector via global average pooling. This tells the network *which channels* matter, but not *where* in the image to look. For traffic signs — which are small, localized objects surrounded by clutter — spatial awareness is critical.

#### The Coordinate Attention Solution

CoordAtt decomposes 2D global pooling into two 1D operations that preserve positional information:

```
Input X: (B, C, H, W)
  │
  ├─── Pool across width ──→ X_h: (B, C, H, 1)    "Where vertically?"
  ├─── Pool across height ─→ X_w: (B, C, 1, W)    "Where horizontally?"
  │
  ├─ Permute X_w → (B, C, W, 1)
  ├─ Concat [X_h, X_w] along dim=2 → (B, C, H+W, 1)
  │
  ├─ Shared 1×1 Conv(C → C_r) + BN + ReLU          "Bottleneck"
  │    └─ C_r = max(8, C // reduction)
  │    └─ Shape: (B, C_r, H+W, 1)
  │
  ├─ Split → Y_h: (B, C_r, H, 1)  +  Y_w: (B, C_r, W, 1)
  ├─ Re-permute Y_w → (B, C_r, 1, W)
  │
  ├─ Conv_h(C_r → C) + Sigmoid → A_h: (B, C, H, 1)    "Vertical attention"
  ├─ Conv_w(C_r → C) + Sigmoid → A_w: (B, C, 1, W)    "Horizontal attention"
  │
  └─ Output = X * A_h * A_w                             "Apply with broadcasting"

Output: (B, C, H, W)  ← same shape as input
```

#### The `reduction` Parameter

The `reduction` argument (default `32`) controls the bottleneck ratio in the shared 1×1 convolution:

```python
mid_channels = max(8, c1 // reduction)
```

- For `c1=512, reduction=32` → `mid_channels = max(8, 16) = 16`
- For `c1=256, reduction=32` → `mid_channels = max(8, 8) = 8`
- The `max(8, ...)` clamp ensures we never create a degenerate 1-or-2-channel bottleneck

A smaller `reduction` (e.g., `16`) gives more capacity but adds parameters. We use `32` to keep the attention blocks lightweight — each CoordAtt adds < 1% to total parameter count.

#### Constructor Arguments

```python
def __init__(self, c1: int, c2: int, reduction: int = 32):
```

| Arg | Source | Description |
|-----|--------|-------------|
| `c1` | Auto-injected by `parse_model` | Input channels — determines pooling and conv sizes |
| `c2` | YAML `args[0]` | Must equal `c1` — enforced by assertion. Attention is shape-preserving |
| `reduction` | YAML `args[1]` or default `32` | Bottleneck reduction ratio |

**`c1 == c2` enforcement:** A runtime assertion `assert c1 == c2` catches misconfigurations immediately. If someone accidentally writes `CoordAtt, [128, 32]` after a layer that outputs 256 channels, the model fails at construction time with a clear error — not silently at training time.

---

## 📐 Deep Dive: `models/configs/yolov8_custom.yaml`

### Why SubPixelConv at Layers 10 and 14

In YOLOv8's Feature Pyramid Network (FPN), the top-down path merges high-level semantic features with low-level spatial features. This requires upsampling the deep maps to match the shallow maps' resolution:

```
Layer 10: SPPF output (P5, 20×20) ──SubPixelConv──→ (40×40) ──Concat──→ with backbone P4 (40×40)
Layer 14: Fused P4    (40×40)     ──SubPixelConv──→ (80×80) ──Concat──→ with backbone P3 (80×80)
```

These are the **only two upsampling points** in the YOLOv8 neck. We replaced both `nn.Upsample` layers with `SubPixelConv` — the exact two locations where spatial detail recovery matters most for small-object detection.

### Why CoordAtt at Layers 13, 17, 21, and 25

CoordAtt is placed **immediately after every `C2f` fusion block** in the neck, right before the features are either:
- Fed to the next stage of the FPN/PAN pathway, or
- Fed directly to the Detect head.

```
Layer 13: After C2f P4 fusion (FPN)  → feeds PAN bottom-up path at layer 19
Layer 17: After C2f P3 fusion (FPN)  → feeds Detect head directly (P3 = small objects!)
Layer 21: After C2f PAN P4 fusion    → feeds Detect head directly (P4 = medium objects)
Layer 25: After C2f PAN P5 fusion    → feeds Detect head directly (P5 = large objects)
```

**Strategic placement logic:** By inserting CoordAtt *after* the C2f fusion blocks but *before* the Detect heads, we allow the attention mechanism to refine the fused features — suppressing weather-induced noise and highlighting sign-shaped spatial patterns — right at the decision boundary. This is where attention has the highest leverage.

### Layer Routing Reference Table

```
Index │ from       │ Module       │ args         │ Role
──────┼────────────┼──────────────┼──────────────┼──────────────────────
  0   │ -1         │ Conv         │ [64,3,2]     │ Backbone stem
  1   │ -1         │ Conv         │ [128,3,2]    │ P2 downsample
  2   │ -1         │ C2f          │ [128,True]   │ P2 features
  3   │ -1         │ Conv         │ [256,3,2]    │ P3 downsample
  4   │ -1         │ C2f          │ [256,True]   │ ★ P3 features → layer 15
  5   │ -1         │ Conv         │ [512,3,2]    │ P4 downsample
  6   │ -1         │ C2f          │ [512,True]   │ ★ P4 features → layer 11
  7   │ -1         │ Conv         │ [1024,3,2]   │ P5 downsample
  8   │ -1         │ C2f          │ [1024,True]  │ P5 features
  9   │ -1         │ SPPF         │ [1024,5]     │ ★ SPPF → layers 10, 23
 10   │ -1         │ SubPixelConv │ [512]        │ ◄ Learnable upsample 2×
 11   │ [-1,6]     │ Concat       │ [1]          │ Merge with P4 skip
 12   │ -1         │ C2f          │ [512]        │ Fuse FPN P4
 13   │ -1         │ CoordAtt     │ [512,32]     │ ◄ Attention FPN P4
 14   │ -1         │ SubPixelConv │ [256]        │ ◄ Learnable upsample 2×
 15   │ [-1,4]     │ Concat       │ [1]          │ Merge with P3 skip
 16   │ -1         │ C2f          │ [256]        │ Fuse FPN P3
 17   │ -1         │ CoordAtt     │ [256,32]     │ ◄ Attention FPN P3 → Detect
 18   │ -1         │ Conv         │ [256,3,2]    │ PAN downsample
 19   │ [-1,13]    │ Concat       │ [1]          │ Merge with FPN P4 attn
 20   │ -1         │ C2f          │ [512]        │ Fuse PAN P4
 21   │ -1         │ CoordAtt     │ [512,32]     │ ◄ Attention PAN P4 → Detect
 22   │ -1         │ Conv         │ [512,3,2]    │ PAN downsample
 23   │ [-1,9]     │ Concat       │ [1]          │ Merge with SPPF
 24   │ -1         │ C2f          │ [1024]       │ Fuse PAN P5
 25   │ -1         │ CoordAtt     │ [1024,32]    │ ◄ Attention PAN P5 → Detect
 26   │ [17,21,25] │ Detect       │ [nc]         │ 3-scale detection
```

---

## 🛠️ Deep Dive: `scripts/phase3_architecture_design.py`

### The `base_modules` Problem

When `parse_model()` iterates over the YAML layers, it checks whether each module class belongs to a hardcoded `base_modules` frozenset:

```python
# Inside ultralytics/nn/tasks.py → parse_model()
base_modules = frozenset({
    Classify, Conv, ConvTranspose, C2f, ..., A2C2f,
})

if m in base_modules:
    c1, c2 = ch[f], args[0]        # ← This is the magic c1/c2 injection
    c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]     # ← Prepends c1, applies width multiplier
```

If our module is **not** in `base_modules`, `parse_model` will call it as `Module(*args)` — passing only the raw YAML args (`[512]` or `[512, 32]`), without the critical `c1` channel and without applying the width multiplier to `c2`. This breaks everything.

### The `inspect` + `exec` Solution

Since `base_modules` is a `frozenset` literal embedded in function source code, we cannot mutate it at runtime (frozensets are immutable, and it's re-created every time the function runs). Instead, we **rewrite the function itself**:

```python
def _patch_parse_model(tasks_module, SubPixelConv, CoordAtt):
    original_fn = tasks_module.parse_model

    # Guard: don't double-patch
    if getattr(original_fn, '_custom_patched', False):
        return

    # 1. Extract source code as a string
    source = inspect.getsource(original_fn)
    source = textwrap.dedent(source)

    # 2. Find the last entry in the base_modules frozenset: "A2C2f,"
    #    and insert our modules right after it
    source = source.replace(
        "A2C2f,",
        "A2C2f,\n            SubPixelConv,\n            CoordAtt,",
        1  # Only replace FIRST occurrence (in base_modules, not repeat_modules)
    )

    # 3. Compile and exec the modified source in the tasks module's globals
    #    This ensures globals()[m] resolves SubPixelConv/CoordAtt (injected in Step 1)
    exec_globals = tasks_module.__dict__
    exec(compile(source, "<patched_parse_model>", "exec"), exec_globals)

    # 4. Mark as patched and replace the module-level function
    patched_fn = exec_globals['parse_model']
    patched_fn._custom_patched = True
    tasks_module.parse_model = patched_fn
```

#### Why This Works

1. `inspect.getsource()` returns the exact Python source of `parse_model` as a string.
2. We do a surgical string replacement: `"A2C2f,"` → `"A2C2f,\n            SubPixelConv,\n            CoordAtt,"`, which adds our two classes to the `base_modules` frozenset literal.
3. `exec()` compiles and runs the modified source inside `tasks_module.__dict__` — the same namespace where the original function lived. This means `globals()[m]` resolves correctly because we already injected `SubPixelConv` and `CoordAtt` into that namespace in Step 1.
4. The `_custom_patched` sentinel prevents double-patching if `register_custom_modules()` is called more than once.

#### Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Ultralytics updates rename `A2C2f` | The script raises `RuntimeError` with a clear message if the marker isn't found |
| Double-patch corruption | `_custom_patched` flag prevents re-execution |
| Namespace pollution | We `exec` into the *existing* module dict, not a new namespace |
| `inspect.getsource()` unavailable (e.g., frozen bytecode) | Extremely unlikely in development — only applies to `.pyc`-only distributions |

---

## 📊 Validation Results

| Metric | Value | Status |
|--------|-------|--------|
| Model build from YAML | Success | ✅ |
| Forward pass `(1, 3, 640, 640)` | No errors | ✅ |
| Detection heads | 3 (P3, P4, P5) | ✅ |
| Number of classes (`nc`) | 143 | ✅ |
| SubPixelConv instances | 2 | ✅ |
| CoordAtt instances | 4 | ✅ |
| Total parameters | ~4.8M | ✅ |
| Trainable parameters | ~4.8M (100%) | ✅ |

Validation report saved to: `results/phase3_architecture_validation.txt`

---

## 🚀 Handoff Instructions for Phase 4 (Enrique — Training)

### How to Use the Custom Architecture

In `train.py` (or any training script), you **must** call `register_custom_modules()` before creating the YOLO model:

```python
"""Phase 4 — Training with custom YOLOv8 architecture."""
import sys
sys.path.insert(0, '.')

# ── STEP 1: Register custom modules into Ultralytics ──
# This MUST be called BEFORE `from ultralytics import YOLO`
from scripts.phase3_architecture_design import register_custom_modules
register_custom_modules()

# ── STEP 2: Create the model from our custom YAML ──
from ultralytics import YOLO
model = YOLO('models/configs/yolov8_custom.yaml')

# ── STEP 3: Train ──
model.train(
    data='data/processed/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,            # GPU index
    project='results',
    name='yolov8_custom_tt100k',
)
```

### Critical Rules

1. **Import order matters.** `register_custom_modules()` must execute *before* `from ultralytics import YOLO`. The function patches `parse_model` at import time.
2. **Run from project root.** Always `cd` to the project root before running scripts, so that `models/` and `scripts/` are importable.
3. **Do not modify `models/custom_modules.py`** unless you also update the YAML args and this report.
4. **The `dataset.yaml` paths** should follow Phase 2's specification:
   ```yaml
   path: ./data
   train: augmented/train/images     # 67,509 augmented images
   val:   split/val/images           # 1,830 original validation images
   test:  split/test/images          # 3,067 original test images
   nc: 143
   ```

### File Inventory

| File | Lines | Role |
|------|-------|------|
| `models/__init__.py` | 0 | Package marker |
| `models/custom_modules.py` | 233 | `SubPixelConv` + `CoordAtt` (pure PyTorch) |
| `models/configs/yolov8_custom.yaml` | 55 | 27-layer architecture definition |
| `scripts/phase3_architecture_design.py` | 240 | Registry injection + validation pipeline |
| `docs/PHASE3_ARCHITECTURE_PLAN.md` | ~430 | Architecture plan (this phase's blueprint) |
| `docs/PHASE3_FINAL_REPORT.md` | — | This document |
| `results/phase3_architecture_validation.txt` | ~15 | Auto-generated validation report |

---

*Phase 3 — Architecture Engineering — COMPLETE ✅*  
*Chief AI Architect | MICAI 2025*
