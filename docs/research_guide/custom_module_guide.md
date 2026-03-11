# A Researcher's Guide to Modifying YOLO Deep Learning Model Architectures

> **Audience:** Researchers and engineers who need to integrate custom layers, blocks, or fusion mechanisms into an existing deep learning framework.
> **Framework focus:** PyTorch (the guide applies to any PyTorch project; framework-specific notes are labeled **[Ultralytics]**).
> **Philosophy:** Understand the system before you change it. One slow methodical step saves ten hours of debugging.

---

## Table of Contents

1. [Mental Model: How a DL Framework is Structured](#1-mental-model-how-a-dl-framework-is-structured)
2. [Phase 1 — Understand the Architecture You Are Modifying](#2-phase-1--understand-the-architecture-you-are-modifying)
3. [Phase 2 — Design Your Custom Module](#3-phase-2--design-your-custom-module)
4. [Phase 3 — Implement and Register the Module](#4-phase-3--implement-and-register-the-module)
5. [Phase 4 — Wire It Into the Model Definition](#5-phase-4--wire-it-into-the-model-definition)
6. [Phase 5 — Sanity Check Before Training](#6-phase-5--sanity-check-before-training)
7. [Phase 6 — Training and Validation](#7-phase-6--training-and-validation)
8. [Common Pitfalls and How to Avoid Them](#8-common-pitfalls-and-how-to-avoid-them)
9. [Checklist](#9-checklist)

---

## 1. Mental Model: How a DL Framework is Structured

Before touching any code, understand that every production DL framework is organised in the same conceptual layers:

```
┌─────────────────────────────────────────────────────────┐
│  Model Config / Blueprint  (YAML, JSON, Python dict)    │  ← describes the architecture
├─────────────────────────────────────────────────────────┤
│  Model Parser / Builder    (parse_model, build_model)   │  ← reads the blueprint
├─────────────────────────────────────────────────────────┤
│  Module Registry           (__init__.py, globals())     │  ← maps string names → classes
├─────────────────────────────────────────────────────────┤
│  Custom Modules            (nn.Module subclasses)       │  ← your actual torch code
├─────────────────────────────────────────────────────────┤
│  Installed Package         (site-packages)              │  ← what Python actually imports
└─────────────────────────────────────────────────────────┘
```

When something breaks, it almost always breaks at exactly one of these layers. Identifying which layer the error belongs to is the first debugging step.

---

## 2. Phase 1 — Understand the Architecture You Are Modifying

### 2.1 Read the Model Blueprint

**[Ultralytics]** Model architectures are defined in YAML files inside `ultralytics/cfg/models/`. Each row in the YAML is one layer:

```yaml
- [from, repeats, module, args]
```

| Field | Meaning |
|---|---|
| `from` | Index of input layer(s). `-1` = previous layer. A list = multi-input. |
| `repeats` | Number of times the module is stacked (depth scaling). |
| `module` | The class name as a string — must match an entry in the registry. |
| `args` | Positional arguments passed to `__init__`, excluding input channels. |

**General PyTorch** projects that do not use YAML typically define the graph in Python code directly (e.g., `nn.Sequential`, calls inside `forward()`). Apply the same reading discipline: trace the data path from input to output.

### 2.2 Trace the Data Flow

Draw or write out the feature map dimensions at every step:

```
Input Image [B, 3, 640, 640]
   → P1 [B, 16, 320, 320]   stride 2
   → P2 [B, 32, 160, 160]   stride 4
   → P3 [B, 64,  80,  80]   stride 8
   → P4 [B, 128, 40,  40]   stride 16
   → P5 [B, 256, 20,  20]   stride 32
```

**Why this matters:** Any fusion layer must receive tensors of compatible spatial size (for addition) or at least the correct channel count (for concatenation). Mismatches here are the #1 source of `RuntimeError: shape mismatch` crashes.

### 2.3 Identify the Existing Fusion Mechanism

Before you can improve it, understand what it currently does:

- Is it `Concat` (channels stack, spatial preserved)?
- Is it element-wise addition (channels unchanged, spatial must match)?
- Is it attention-weighted (e.g., SE, CBAM)?

Read the existing module's `forward()` method carefully.

---

## 3. Phase 2 — Design Your Custom Module

Answer these questions **on paper** before writing a single line of code:

### 3.1 What are the inputs and outputs?

| Question | Example Answer (BiFPN_Concat) |
|---|---|
| How many input tensors? | 2 (from two different backbone/neck levels) |
| Input shapes? | Each `[B, C, H, W]` — same H, W but same C too |
| Output shape? | `[B, C*2, H, W]` — concatenated channels |
| Is output shape same as a plain `Concat`? | Yes — drop-in compatible |

### 3.2 What learnable parameters does it introduce?

| Parameter | Type | Shape | Init value |
|---|---|---|---|
| Fusion weights `w` | `nn.Parameter` | `(n_inputs,)` | `torch.ones(n)` for equal priority |

### 3.3 What is the forward computation?

Write out the math before turning it into code:

$$O = \text{cat}\left(\frac{w_0}{\sum w + \varepsilon} \cdot x_0, \; \frac{w_1}{\sum w + \varepsilon} \cdot x_1\right)$$

---

## 4. Phase 3 — Implement and Register the Module

### 4.1 Writing the `nn.Module`

Every custom module follows the same skeleton:

```python
import torch
import torch.nn as nn

class MyCustomLayer(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()                          # ALWAYS call super().__init__()
        # Define sub-layers and learnable params here
        self.w = nn.Parameter(torch.ones(param1))  # learnable weight
        self.sublayer = nn.Conv2d(...)              # any nn.Module is tracked
        self.epsilon = 1e-4                         # non-learnable constant

    def forward(self, x):
        # Define the computation graph here
        # Everything you write here is differentiable
        w = torch.relu(self.w)                     # clamp weights positive
        ...
        return output
```

**Key rules:**
- All `nn.Module` children and `nn.Parameter` objects defined in `__init__` are automatically registered. You do not need to manually add them to an optimizer — PyTorch handles this.
- `forward()` is called when you do `layer(x)`. Never call `forward()` directly.
- Never store mutable state (e.g., batch-dependent values) as instance attributes inside `forward()` — this breaks multi-GPU training.

### 4.2 `nn.Parameter` — The Learnable Weight Mechanism

`nn.Parameter` is a Tensor wrapper that:
1. Registers the tensor with the module so it appears in `model.parameters()`
2. Sets `requires_grad=True` automatically

```python
# Correct — learnable
self.w = nn.Parameter(torch.ones(2))

# Wrong — not tracked, optimizer ignores it
self.w = torch.ones(2)
```

**Initialization:** For scale/fusion weights, initialize to `torch.ones(n)` so all inputs start with equal importance. Kaiming/Xavier initialization is designed for weight matrices in convolutions — do not use it for scalar weight vectors.

### 4.3 Registering the Module

**[Ultralytics]** You must add entries in exactly two files:

**File 1: `ultralytics/nn/modules/__init__.py`**
```python
# Add to the from .conv import (...) line
from .conv import (BiFPN_Concat, CBAM, ChannelAttention, Concat, ...)

# Add to __all__
__all__ = (
    "BiFPN_Concat",
    "CBAM",
    "Concat",
    ...
)
```

**File 2: `ultralytics/nn/tasks.py`**
```python
# 1. Add to the import block at the top
from ultralytics.nn.modules import (BiFPN_Concat, Concat, ...)

# 2. Add to the correct elif branch in parse_model() — do NOT add to base_modules
# if your layer takes multiple inputs (f is a list):
elif m in {Concat, BiFPN_Concat}:
    c2 = sum(ch[x] for x in f)
```

**General PyTorch:** If the framework uses `globals()` to resolve string names, just ensure the class is in scope where `globals()` is called. If it uses a registry dict, add an entry there.

### 4.4 Understanding `base_modules` vs Multi-input Branches

**[Ultralytics]** The `parse_model` function separates modules into two categories:

| Category | `f` (from) type | Channel logic | Examples |
|---|---|---|---|
| `base_modules` | Single integer | `c1, c2 = ch[f], args[0]` | `Conv`, `C2f`, `SPPF` |
| Multi-input branch | List of integers | `c2 = sum(ch[x] for x in f)` | `Concat`, `BiFPN_Concat` |

**If you add a multi-input layer to `base_modules` by mistake**, you will get:
```
TypeError: list indices must be integers or slices, not list
```
because the parser tries to do `ch[[-1, 6]]` instead of `ch[-1] + ch[6]`.

---

## 5. Phase 4 — Wire It Into the Model Definition

**[Ultralytics YAML]** Replace the target module name in the YAML:

```yaml
# Before
- [[-1, 6], 1, Concat, [1]]

# After — n=2 inputs, dimension=1
- [[-1, 6], 1, BiFPN_Concat, [2, 1]]
```

**Indexing rules:**
- Layer indices in YAML are 0-based and assigned in order starting from the first backbone layer.
- Negative index `-1` always means "the immediately preceding layer."
- Count skeleton layers out carefully and annotate each block with `# layer_index`.

**General PyTorch:** In code-defined models, instantiate your module in `__init__` and call it in `forward()`:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bifpn = BiFPN_Concat(n=2)

    def forward(self, features):
        p3, p4 = features
        return self.bifpn([p3, p4])
```

---

## 6. Phase 5 — Sanity Check Before Training

Never start a training run on a model you haven't verified. Run these checks in order:

### 6.1 Build Check

```python
from ultralytics import YOLO
model = YOLO("path/to/your_model.yaml")
model.info()
```

Expected output: layer table + `N layers, M parameters, M gradients, G GFLOPs` with no errors.

**Alternatively (pure PyTorch):**
```python
import torch
model = MyModel()
x = torch.randn(1, 3, 640, 640)
out = model(x)
print(out.shape)  # should match your expected output shape
```

### 6.2 Gradient Check

Confirm your new parameters appear in the optimizer:

```python
for name, param in model.named_parameters():
    if "bifpn" in name.lower() or "w" in name:
        print(name, param.shape, param.requires_grad)
```

All new learnable weights must show `requires_grad=True`.

### 6.3 Device Check

Ensure the model and a dummy input run on GPU without error:

```python
model = model.cuda()
x = torch.randn(1, 3, 640, 640).cuda()
out = model(x)
```

---

## 7. Phase 6 — Training and Validation

### 7.1 Quick Smoke Test

Before committing to a full dataset, run for just 1-3 epochs on a tiny dataset to ensure the loss decreases and no runtime errors occur. Ultralytics provides `coco8` — 8 images, downloads automatically (~1 MB):

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/your_custom_model.yaml")

model.train(
    data="coco8.yaml",    # 8-image dataset — just for sanity
    epochs=3,
    imgsz=640,
    batch=4,
    device=0,             # use GPU 0, or 'cpu'
    workers=0,            # set to 0 on Windows to avoid DataLoader issues
    plots=False,          # skip plots for speed
)
```

**What to look for:**
- `loss` values decrease over epochs ✅
- No `NaN` loss (if you see this, your weight initialization or learning rate is likely wrong) ✅
- No shape mismatch errors ✅

### 7.2 Full Training Run

Once the smoke test passes, switch to your actual dataset:

```python
model.train(
    data="path/to/your_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    project="runs/detect",
    name="yolov8n-p2-BiFPN",
    exist_ok=False,
)
```

### 7.3 Comparing Against the Baseline

For research, you must establish a **fair baseline**:

1. Train the **original** model (`yolov8-p2.yaml`) on the same dataset with the same hyperparameters.
2. Train your **modified** model (`yolov8-p2-BiFPN.yaml`) identically.
3. Compare `mAP@50` and `mAP@50-95` from the `results.csv` files.

```python
# Validate and print metrics
results = model.val(data="your_dataset.yaml")
print(results.box.map)    # mAP@50-95
print(results.box.map50)  # mAP@50
```

---

## 8. Common Pitfalls and How to Avoid Them

### 8.1 Python Imports from Site-Packages Instead of Your Local Files

**Symptom:** `ImportError: cannot import name 'MyModule' from 'package.nn.modules'`

**Cause:** The installed pip package in `.venv/Lib/site-packages/` takes priority over your local files.

**Fix:** Install the local package in editable mode:
```bash
pip install -e .
```
This creates a `.pth` redirect so Python uses your local source files directly.

### 8.2 Adding Multi-Input Layers to `base_modules`

**Symptom:** `TypeError: list indices must be integers or slices, not list`

**Cause:** `base_modules` assumes a single-input layer. Multi-input layers need their own channel arithmetic branch.

**Fix:** Add a dedicated `elif` branch in `parse_model()` as shown in Phase 3.4.

### 8.3 Forgetting `self.eps` in the Denominator

**Symptom:** `RuntimeError: invalid value encountered` or silent `NaN` loss.

**Cause:** Divide-by-zero if all learned weights are driven to 0 during training.

**Fix:** Always include `+ self.eps` in the denominator of any normalized fusion:
```python
weight_sum = torch.sum(w) + self.eps
```

### 8.4 Using `torch.zeros_like(list)` Instead of a Python List

**Symptom:** `TypeError: expected Tensor, got list`

**Cause:** `torch.zeros_like(x)` requires a Tensor. If `x` is a Python list of Tensors, you must initialize a Python list instead.

**Fix:**
```python
# Wrong
output = torch.zeros_like(x)

# Correct
output = []
for i, xi in enumerate(x):
    output.append(w[i] * xi / weight_sum)
return torch.cat(output, dim=self.d)
```

### 8.5 Layer Index Drift in YAML

**Symptom:** Wrong feature maps being fused — the model trains but performance is poor.

**Cause:** After inserting or removing layers, the absolute layer indices in `from` fields shift.

**Fix:** Always annotate every layer in the YAML with its index:
```yaml
- [-1, 3, C2f, [512]] # 12
```
After any insertion, re-count and update all downstream absolute indices.

### 8.6 `nn.Parameter` Registered but Not Moved to GPU

**Symptom:** `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Parameters defined with `nn.Parameter` are automatically moved by `.to(device)` and `.cuda()`. If you instead use a raw `torch.Tensor` (not wrapped), it stays on CPU.

**Fix:** Always use `nn.Parameter` for learnable tensors, never raw tensors.

---

## 9. Checklist

Use this checklist every time you add a custom module to a PyTorch / Ultralytics project.

### Module Implementation
- [ ] Class inherits from `nn.Module`
- [ ] `super().__init__()` called first in `__init__`
- [ ] All learnable tensors wrapped in `nn.Parameter`
- [ ] `self.eps` included in any normalized division
- [ ] `forward()` returns the correct shape — verified on paper before coding

### Registration
- [ ] Module imported and added to `__all__` in `nn/modules/__init__.py`
- [ ] Module imported in `tasks.py` (or equivalent model builder)
- [ ] Module placed in the correct parsing branch (`base_modules` for single-input, dedicated `elif` for multi-input)

### Model Config
- [ ] YAML (or model definition) updated to reference the new module by exact class name
- [ ] `args` in YAML match the `__init__` signature (excluding `c1` — that is injected by the parser)
- [ ] Layer indices in `from` fields verified by counting from the top of the YAML

### Verification
- [ ] `pip install -e .` run so local files are used
- [ ] `model.info()` or `model(dummy_input)` runs without error
- [ ] Gradient check: new parameters show `requires_grad=True`
- [ ] Smoke train: 1-3 epochs, loss decreases, no NaN

### Research Rigor
- [ ] Baseline trained on identical dataset + hyperparameters
- [ ] Only one variable changed at a time (your module vs. original)
- [ ] Results reported as `mAP@50` and `mAP@50-95` on a held-out test set
