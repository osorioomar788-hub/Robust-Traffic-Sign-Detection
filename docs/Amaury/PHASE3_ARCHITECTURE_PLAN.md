# Phase 3 — Architecture Engineering: Master Execution Plan
## Robust Traffic Sign Detection | 25th MICAI
**Architect:** Chief AI Architect (Planner Mode)  
**Executor (Lead):** Amaury (Phase 3)  
**Upstream Dependency:** Phase 2 complete — Robust-TT100K dataset (67,509 train / 1,830 val / 3,067 test — 143 classes)  
**Base Model:** YOLOv8n (`yolov8n.pt`)

---

## Table of Contents
1. [Objective](#1-objective)
2. [Mathematical Blueprint](#2-mathematical-blueprint)
3. [File Structure Plan](#3-file-structure-plan)
4. [Integration Strategy — Ultralytics Registry Injection](#4-integration-strategy--ultralytics-registry-injection)
5. [Custom YOLOv8 YAML Architecture](#5-custom-yolov8-yaml-architecture)
6. [Execution Steps — Coder Agent Checklist](#6-execution-steps--coder-agent-checklist)
7. [Verification Plan](#7-verification-plan)
8. [NEXT_PROMPT](#8-next_prompt)

---

## 1. Objective

Enhance the YOLOv8n architecture with two custom modules targeting the project's core challenges:

| Challenge | Module | Rationale |
|-----------|--------|-----------|
| Small traffic signs (< 32×32 px on 2048×2048 TT100K images) | **Sub-Pixel Convolution (ESPCN)** | Learnable upsampling replaces naïve `nn.Upsample` — recovers fine-grained spatial detail lost in deep feature maps |
| Adverse weather (rain, fog, night) obscures discriminative regions | **Coordinate Attention** | Encodes directional long-range dependencies — helps the network focus on sign regions even under degraded contrast |

The modifications must be **non-invasive**: we will NOT modify Ultralytics source code. All customizations live in our project files and are injected at runtime via Python's module registry.

---

## 2. Mathematical Blueprint

### 2.1 Sub-Pixel Convolution (`SubPixelConv`)

> Reference: Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network", CVPR 2016.

#### 2.1.1 Core Operation — PixelShuffle

Given an input tensor of shape `(B, C_in, H, W)` and an upscale factor `r`:

```
Step 1: Convolution         (B, C_in, H, W) → (B, C_out · r², H, W)
Step 2: PixelShuffle        (B, C_out · r², H, W) → (B, C_out, H·r, W·r)
Step 3: BatchNorm + SiLU    (B, C_out, H·r, W·r) → (B, C_out, H·r, W·r)  [no shape change]
```

#### 2.1.2 Concrete Example (YOLOv8n Upsampling Replacement)

In YOLOv8n, the neck uses `nn.Upsample(scale_factor=2)` to upsample feature maps. We replace these with `SubPixelConv`:

```
Input:  (B, 256, 20, 20)  — from a backbone feature level (P4)

Step 1: Conv2d(256, 128 * 2², kernel=3, pad=1)
        (B, 256, 20, 20) → (B, 512, 20, 20)

Step 2: nn.PixelShuffle(upscale_factor=2)
        (B, 512, 20, 20) → (B, 128, 40, 40)

Step 3: nn.BatchNorm2d(128) + nn.SiLU()
        (B, 128, 40, 40) → (B, 128, 40, 40)

Output: (B, 128, 40, 40)   — matches P3 spatial dimensions for concatenation
```

#### 2.1.3 PixelShuffle Tensor Rearrangement (Formal)

For an input `X ∈ ℝ^(B × C·r² × H × W)`, the output `Y ∈ ℝ^(B × C × rH × rW)` is computed as:

```
Y[b, c, h·r + δh, w·r + δw] = X[b, c·r² + δh·r + δw, h, w]

where:
  b ∈ [0, B)           — batch index
  c ∈ [0, C)           — output channel index
  h ∈ [0, H), w ∈ [0, W)  — spatial position in input
  δh ∈ [0, r), δw ∈ [0, r) — sub-pixel offset
```

#### 2.1.4 PyTorch Class Signature

```python
class SubPixelConv(nn.Module):
    """Learnable upsampling via sub-pixel convolution (ESPCN).
    
    Replaces nn.Upsample in YOLOv8 neck for improved small-object detail recovery.
    """
    def __init__(self, c1: int, c2: int, scale: int = 2):
        """
        Args:
            c1: Input channels  (will be provided by parse_model from prev layer)
            c2: Output channels (specified in YAML args)
            scale: Upscale factor (default 2, matching YOLOv8 upsample behavior)
        """
        # Conv2d: c1 → c2 * scale², kernel=3, pad=1
        # PixelShuffle(scale)
        # BatchNorm2d(c2)
        # SiLU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, c1, H, W) → return: (B, c2, H*scale, W*scale)
```

> **CRITICAL for `parse_model` compatibility:** The `__init__` signature MUST accept `c1` as the first argument (input channels from previous layer) and `c2` as the second (output channels from YAML args). This is how Ultralytics' `parse_model` wires layers together.

---

### 2.2 Coordinate Attention (`CoordAtt`)

> Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021.

#### 2.2.1 Full Pipeline — Step by Step

Given an input tensor `X` of shape `(B, C, H, W)`:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: Coordinate Information Embedding (Dual Directional Pooling)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1a — Horizontal Global Average Pool (pool across width):
  X_h = AdaptiveAvgPool2d((H, 1))(X)
  (B, C, H, W) → (B, C, H, 1)

Step 1b — Vertical Global Average Pool (pool across height):
  X_w = AdaptiveAvgPool2d((1, W))(X)
  (B, C, H, W) → (B, C, 1, W)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: Concatenation + Shared Transform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 2a — Permute X_w for concatenation:
  X_w_T = X_w.permute(0, 1, 3, 2)
  (B, C, 1, W) → (B, C, W, 1)

Step 2b — Concatenate along spatial dimension (dim=2):
  Y = torch.cat([X_h, X_w_T], dim=2)
  cat[(B, C, H, 1), (B, C, W, 1)] → (B, C, H+W, 1)

Step 2c — Shared 1×1 Conv + BatchNorm + ReLU (reduce channels):
  C_r = max(8, C // reduction)   [reduction = 32 default]
  Y = ReLU(BN(Conv2d(C, C_r, kernel=1)(Y)))
  (B, C, H+W, 1) → (B, C_r, H+W, 1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3: Split + Independent Attention Maps
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 3a — Split back along dim=2:
  Y_h, Y_w = Y.split([H, W], dim=2)
  (B, C_r, H+W, 1) → (B, C_r, H, 1) + (B, C_r, W, 1)

Step 3b — Re-permute Y_w:
  Y_w = Y_w.permute(0, 1, 3, 2)
  (B, C_r, W, 1) → (B, C_r, 1, W)

Step 3c — Independent 1×1 Convolutions + Sigmoid:
  A_h = Sigmoid(Conv2d(C_r, C, kernel=1)(Y_h))
  (B, C_r, H, 1) → (B, C, H, 1)

  A_w = Sigmoid(Conv2d(C_r, C, kernel=1)(Y_w))
  (B, C_r, 1, W) → (B, C, 1, W)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 4: Apply Attention (Element-wise Multiplication)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 4 — Element-wise multiply with broadcasting:
  Out = X * A_h * A_w
  (B, C, H, W) * (B, C, H, 1) * (B, C, 1, W) → (B, C, H, W)
```

#### 2.2.2 Concrete Example (Inserted after C2f block)

```
Input:  (B, 128, 40, 40)  — output of a C2f block in the neck

Stage 1:
  X_h: (B, 128, 40, 1)  — pool across W=40
  X_w: (B, 128, 1, 40)  — pool across H=40

Stage 2:
  X_w_T:    (B, 128, 40, 1)
  Y concat: (B, 128, 80, 1)     — H+W = 40+40
  C_r = max(8, 128//32) = 4 → clamped to 8
  Y_reduced: (B, 8, 80, 1)

Stage 3:
  Y_h: (B, 8, 40, 1) → A_h: (B, 128, 40, 1)
  Y_w: (B, 8, 40, 1) → (B, 8, 1, 40) → A_w: (B, 128, 1, 40)

Stage 4:
  Out = X * A_h * A_w: (B, 128, 40, 40)

Output: (B, 128, 40, 40)  — SAME shape as input (attention is a refinement, not a transform)
```

#### 2.2.3 PyTorch Class Signature

```python
class CoordAtt(nn.Module):
    """Coordinate Attention block (Hou et al., CVPR 2021).
    
    Encodes channel + positional information via directional pooling.
    Inserted after C2f blocks to enhance focus on sign regions.
    """
    def __init__(self, c1: int, c2: int, reduction: int = 32):
        """
        Args:
            c1: Input channels (from previous layer, auto-injected by parse_model)
            c2: Output channels (must equal c1 — attention does not change channels)
            reduction: Channel reduction ratio for the shared transform
        """
        # NOTE: c2 is accepted for parse_model compat but MUST equal c1
        # AdaptiveAvgPool2d for (H,1) and (1,W)
        # Shared Conv2d(c1, max(8, c1//reduction), 1) + BN + ReLU
        # conv_h: Conv2d(max(8, c1//reduction), c1, 1)
        # conv_w: Conv2d(max(8, c1//reduction), c1, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, c1, H, W) → return: (B, c1, H, W)  [same shape]
```

> **CRITICAL:** `CoordAtt` is shape-preserving. In the YAML, `c1 == c2`. This means when placed after a `C2f` block, the channel count is NOT altered. `parse_model` will pass the previous layer's output channels as `c1`.

---

## 3. File Structure Plan

### 3.1 Files to CREATE

| File | Purpose |
|------|---------|
| `models/__init__.py` | Empty `__init__` to make `models/` a Python package |
| `models/custom_modules.py` | Contains `SubPixelConv` and `CoordAtt` classes |
| `models/configs/yolov8_custom.yaml` | Modified YOLOv8n architecture YAML referencing custom modules |
| `scripts/phase3_architecture_design.py` | Entry-point script: registers modules, builds model, runs validation |
| `docs/PHASE3_ARCHITECTURE_PLAN.md` | This document (already created) |

### 3.2 Files to MODIFY

| File | Modification |
|------|-------------|
| `requirements.txt` | Add `ultralytics>=8.0.0` and `torch>=2.0.0` / `torchvision>=0.15.0` |
| `train.py` | Replace placeholder with actual training invocation using custom YAML |

### 3.3 Detailed File Contents Plan

#### `models/custom_modules.py`

```
models/custom_modules.py
├── import torch, torch.nn
├── class SubPixelConv(nn.Module)
│   ├── __init__(self, c1, c2, scale=2)
│   └── forward(self, x) → Tensor
├── class CoordAtt(nn.Module)
│   ├── __init__(self, c1, c2, reduction=32)
│   └── forward(self, x) → Tensor
└── # NO module registration here — that happens in the entry script
```

#### `models/configs/yolov8_custom.yaml`

```
# Modified from yolov8n.yaml
# Changes:
#   1. nn.Upsample layers replaced with SubPixelConv
#   2. CoordAtt blocks inserted after key C2f blocks in the neck
# See Section 5 for the full YAML specification.
```

#### `scripts/phase3_architecture_design.py`

```
scripts/phase3_architecture_design.py
├── import sys, torch, ultralytics
├── Register custom modules into ultralytics.nn.modules namespace
├── Build model from custom YAML
├── Run shape validation with dummy input (1, 3, 640, 640)
├── Print layer-by-layer summary
└── Export architecture diagram (optional)
```

---

## 4. Integration Strategy — Ultralytics Registry Injection

### 4.1 The Problem

Ultralytics' `parse_model()` in `ultralytics/nn/tasks.py` resolves layer/module names by looking them up in a known namespace. If it encounters `SubPixelConv` in the YAML and can't find it, it throws a `ValueError`. We need to make our custom classes discoverable **without modifying any Ultralytics source files**.

### 4.2 The Solution — Runtime Module Injection

The `parse_model` function resolves module names via Python's standard `eval()` or `getattr()` from a set of known module namespaces including:
- `ultralytics.nn.modules`
- `torch.nn`

**Our injection strategy:** Before instantiating the YOLO model, we programmatically inject our classes into the `ultralytics.nn.modules` namespace:

```python
# === REGISTRY INJECTION — Phase 3 Integration Strategy ===
import ultralytics.nn.modules as ultralytics_modules
from models.custom_modules import SubPixelConv, CoordAtt

# Inject into Ultralytics' module namespace
setattr(ultralytics_modules, 'SubPixelConv', SubPixelConv)
setattr(ultralytics_modules, 'CoordAtt', CoordAtt)

# ALSO inject into the tasks module's global namespace for eval() resolution
import ultralytics.nn.tasks as ultralytics_tasks
ultralytics_tasks.SubPixelConv = SubPixelConv
ultralytics_tasks.CoordAtt = CoordAtt

# Now the YAML parser will find our custom modules
from ultralytics import YOLO
model = YOLO('models/configs/yolov8_custom.yaml')
```

### 4.3 Why This Works

1. `parse_model()` iterates over each `[from, repeats, module_name, args]` entry in the YAML.
2. For each `module_name`, it does approximately:
   ```python
   m = getattr(ultralytics.nn.modules, module_name, None) or eval(module_name)
   ```
3. By injecting via `setattr`, our classes exist in the namespace at lookup time.
4. `parse_model` then calls `m(c1, *args)` — which is why our `__init__` signatures must accept `c1` as the first positional argument.

### 4.4 Safety Checklist

- [x] No Ultralytics source files modified
- [x] Injection happens BEFORE `YOLO()` constructor is called
- [x] Injection is idempotent (safe to call multiple times)
- [x] Custom module `__init__` signatures match `parse_model` calling convention: `Module(c1, *yaml_args)`
- [x] Works with any Ultralytics version `>=8.0.0`

---

## 5. Custom YOLOv8 YAML Architecture

### 5.1 Reference: Vanilla YOLOv8n Backbone (UNCHANGED)

The backbone remains identical to the stock `yolov8n.yaml`. No modifications:

```yaml
# Ultralytics YOLOv8n backbone — DO NOT MODIFY
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]          # 0  — P1/2    (B, 64, 320, 320)
  - [-1, 1, Conv, [128, 3, 2]]         # 1  — P2/4    (B, 128, 160, 160)
  - [-1, 3, C2f, [128, True]]          # 2             (B, 128, 160, 160)
  - [-1, 1, Conv, [256, 3, 2]]         # 3  — P3/8    (B, 256, 80, 80)
  - [-1, 6, C2f, [256, True]]          # 4             (B, 256, 80, 80)
  - [-1, 1, Conv, [512, 3, 2]]         # 5  — P4/16   (B, 512, 40, 40)
  - [-1, 6, C2f, [512, True]]          # 6             (B, 512, 40, 40)
  - [-1, 1, Conv, [1024, 3, 2]]        # 7  — P5/32   (B, 1024, 20, 20)
  - [-1, 3, C2f, [1024, True]]         # 8             (B, 1024, 20, 20)
  - [-1, 1, SPPF, [1024, 5]]           # 9             (B, 1024, 20, 20)
```

### 5.2 Modified Head (Neck + Detection)

This is where we inject our custom modules. Changes from vanilla `yolov8n.yaml` are marked with `# ◄ CUSTOM`:

```yaml
head:
  # ─── Top-down path (FPN) ─────────────────────────────────────
  - [-1, 1, SubPixelConv, [512]]        # 10  (B,1024,20,20)→(B,512,40,40)  ◄ CUSTOM replaces nn.Upsample
  - [[-1, 6], 1, Concat, [1]]           # 11  cat with backbone P4 → (B,1024,40,40)
  - [-1, 3, C2f, [512]]                 # 12  (B,1024,40,40)→(B,512,40,40)
  - [-1, 1, CoordAtt, [512, 32]]        # 13  (B,512,40,40)→(B,512,40,40)   ◄ CUSTOM attention

  - [-1, 1, SubPixelConv, [256]]        # 14  (B,512,40,40)→(B,256,80,80)   ◄ CUSTOM replaces nn.Upsample
  - [[-1, 4], 1, Concat, [1]]           # 15  cat with backbone P3 → (B,512,80,80)
  - [-1, 3, C2f, [256]]                 # 16  (B,512,80,80)→(B,256,80,80)
  - [-1, 1, CoordAtt, [256, 32]]        # 17  (B,256,80,80)→(B,256,80,80)   ◄ CUSTOM attention

  # ─── Bottom-up path (PAN) ────────────────────────────────────
  - [-1, 1, Conv, [256, 3, 2]]          # 18  (B,256,80,80)→(B,256,40,40)   downsample
  - [[-1, 13], 1, Concat, [1]]          # 19  cat with layer 13 → (B,768,40,40)
  - [-1, 3, C2f, [512]]                 # 20  (B,768,40,40)→(B,512,40,40)
  - [-1, 1, CoordAtt, [512, 32]]        # 21  (B,512,40,40)→(B,512,40,40)   ◄ CUSTOM attention

  - [-1, 1, Conv, [512, 3, 2]]          # 22  (B,512,40,40)→(B,512,20,20)   downsample
  - [[-1, 9], 1, Concat, [1]]           # 23  cat with backbone SPPF → (B,1536,20,20)
  - [-1, 3, C2f, [1024]]               # 24  (B,1536,20,20)→(B,1024,20,20)
  - [-1, 1, CoordAtt, [1024, 32]]       # 25  (B,1024,20,20)→(B,1024,20,20) ◄ CUSTOM attention

  # ─── Detection heads ─────────────────────────────────────────
  - [[17, 21, 25], 1, Detect, [nc]]     # 26  Detect(P3=256, P4=512, P5=1024)
```

### 5.3 Summary of Architectural Changes

| Layer Index | Vanilla YOLOv8n | Custom (Ours) | Purpose |
|-------------|-----------------|---------------|---------|
| 10 | `nn.Upsample(scale=2)` | `SubPixelConv(512)` | Learnable 2× upsampling with spatial detail |
| 13 | *(does not exist)* | `CoordAtt(512, 32)` | Position-aware attention after P4 fusion |
| 14 | `nn.Upsample(scale=2)` | `SubPixelConv(256)` | Learnable 2× upsampling with spatial detail |
| 17 | *(does not exist)* | `CoordAtt(256, 32)` | Position-aware attention after P3 fusion |
| 21 | *(does not exist)* | `CoordAtt(512, 32)` | Position-aware attention after PAN P4 |
| 25 | *(does not exist)* | `CoordAtt(1024, 32)` | Position-aware attention after PAN P5 |

**Net effect:** +2 SubPixelConv, +4 CoordAtt = 6 custom layers added. Detect head indices shift accordingly.

---

## 6. Execution Steps — Coder Agent Checklist

> Each step below is designed to be executed sequentially by a Coder Agent. Each step should be committed independently with a descriptive commit message.

### Step 1: Create `models/custom_modules.py`
- [ ] Create `models/__init__.py` (empty file)
- [ ] Create `models/custom_modules.py`
- [ ] Implement `SubPixelConv(nn.Module)` with exact signature: `__init__(self, c1, c2, scale=2)`
  - Use `nn.Conv2d(c1, c2 * scale**2, kernel_size=3, padding=1, bias=False)`
  - Use `nn.PixelShuffle(scale)`
  - Use `nn.BatchNorm2d(c2)` + `nn.SiLU()`
- [ ] Implement `CoordAtt(nn.Module)` with exact signature: `__init__(self, c1, c2, reduction=32)`
  - Implement dual-direction pooling: `AdaptiveAvgPool2d((H, 1))` and `AdaptiveAvgPool2d((1, W))`
  - Shared transform: `Conv2d(c1, max(8, c1//reduction), 1)` + `BatchNorm2d` + `ReLU`
  - Split + independent 1×1 convs back to `c1` channels + `Sigmoid`
  - Element-wise multiply: `x * a_h * a_w`
- [ ] Add module-level docstrings and inline comments matching the math in Section 2

### Step 2: Create `models/configs/yolov8_custom.yaml`
- [ ] Create `models/configs/` directory
- [ ] Write the YAML file exactly as specified in Section 5.2
- [ ] Include header comments with metadata: `# YOLOv8-Custom | Phase 3 | MICAI 2025`
- [ ] Set `nc: 143` and `scales` section matching `yolov8n` scale (`n: [0.33, 0.25, 1024]`)

### Step 3: Create `scripts/phase3_architecture_design.py`
- [ ] Implement the Ultralytics registry injection as specified in Section 4.2
- [ ] Build model from YAML: `model = YOLO('models/configs/yolov8_custom.yaml')`
- [ ] Run forward pass with dummy input `torch.randn(1, 3, 640, 640)` → verify no errors
- [ ] Print model summary: `model.model.info(verbose=True)`
- [ ] Verify each detection head's input matches expected channels `[256, 512, 1024]`
- [ ] Save architecture validation report to `results/phase3_architecture_validation.txt`
- [ ] Print `[PHASE 3 COMPLETE] Architecture validated successfully.`

### Step 4: Update `requirements.txt`
- [ ] Add `ultralytics>=8.0.0` if not already present
- [ ] Add `torch>=2.0.0` if not already present
- [ ] Add `torchvision>=0.15.0` if not already present

### Step 5: Update `train.py`
- [ ] Replace placeholder content with:
  ```python
  """Phase 4 entry point — uses Phase 3 custom architecture."""
  import sys
  sys.path.insert(0, '.')
  from scripts.phase3_architecture_design import register_custom_modules
  register_custom_modules()
  from ultralytics import YOLO
  model = YOLO('models/configs/yolov8_custom.yaml')
  model.train(data='data/processed/dataset.yaml', epochs=100, imgsz=640)
  ```

---

## 7. Verification Plan

### 7.1 Automated Shape Validation (scripts/phase3_architecture_design.py)

The Phase 3 script itself IS the verification. When executed:

```bash
# From project root, with venv activated:
python scripts/phase3_architecture_design.py
```

**Expected outputs (all must pass):**

| Check | Expected Result |
|-------|----------------|
| Model builds from YAML | No `ValueError` or `KeyError` |
| Forward pass `(1,3,640,640)` | No runtime shape mismatch errors |
| Detection head P3 channels | 256 |
| Detection head P4 channels | 512 |
| Detection head P5 channels | 1024 |
| Total parameters | ~3.5M–4.5M (slightly more than vanilla 3.2M due to SubPixelConv/CoordAtt) |

### 7.2 Unit-Level Tensor Shape Tests

Add to `scripts/phase3_architecture_design.py` or as a separate test:

```python
# SubPixelConv shape test
m = SubPixelConv(256, 128, scale=2)
x = torch.randn(2, 256, 20, 20)
assert m(x).shape == (2, 128, 40, 40), "SubPixelConv shape mismatch!"

# CoordAtt shape test  
m = CoordAtt(128, 128, reduction=32)
x = torch.randn(2, 128, 40, 40) 
assert m(x).shape == (2, 128, 40, 40), "CoordAtt shape mismatch!"

# CoordAtt with non-square input
x = torch.randn(2, 64, 80, 120)
m = CoordAtt(64, 64, reduction=32)
assert m(x).shape == (2, 64, 80, 120), "CoordAtt non-square shape mismatch!"
```

### 7.3 Manual Verification

After running the script, the user should verify:
1. The printed model summary shows `SubPixelConv` and `CoordAtt` layers at the expected positions
2. No warnings about unrecognized modules in Ultralytics output
3. The parameter count is in the expected range (~3.5M–4.5M for yolov8n-scale)

---

## 8. NEXT_PROMPT

Copy-paste this prompt to summon the first Coder Agent and execute **Step 1**:

---

```
[NEXT_PROMPT]

# SYSTEM BEHAVIOR
You are a Senior PyTorch Engineer executing Phase 3, Step 1 of the MICAI project.
Your ONLY job is to write production-quality code. Follow the plan EXACTLY.

<context>
Read the architecture plan: `docs/PHASE3_ARCHITECTURE_PLAN.md`
Focus on Section 2 (Mathematical Blueprint) and Section 6, Step 1.
</context>

<task>
1. Create `models/__init__.py` (empty file to make it a Python package).
2. Create `models/custom_modules.py` containing:
   - `SubPixelConv(nn.Module)`: Learnable upsampling via sub-pixel convolution.
     - Signature: `__init__(self, c1: int, c2: int, scale: int = 2)`
     - Implementation: Conv2d(c1, c2*scale², k=3, pad=1) → PixelShuffle(scale) → BN(c2) → SiLU()
   - `CoordAtt(nn.Module)`: Coordinate Attention block.
     - Signature: `__init__(self, c1: int, c2: int, reduction: int = 32)`
     - Implementation: Dual 1D pooling → Concat → Shared Conv → Split → Independent Conv + Sigmoid → Multiply
3. After writing the code, run inline unit tests to verify tensor shapes:
   - SubPixelConv(256, 128, 2): input (2,256,20,20) → output (2,128,40,40)
   - CoordAtt(128, 128, 32):   input (2,128,40,40) → output (2,128,40,40)
</task>

<constraints>
- Use EXACTLY the `__init__` signatures specified (c1 first, c2 second) — this is required for Ultralytics parse_model compatibility.
- Include comprehensive docstrings and inline comments referencing the paper.
- Use type hints throughout.
- Do NOT import or reference Ultralytics in this file — keep it pure PyTorch.
- After creating the files, run a quick Python test to validate the shapes pass.
</constraints>
```

---

*End of Phase 3 Architecture Plan.*
*Document created by Chief AI Architect — Planner Mode.*
