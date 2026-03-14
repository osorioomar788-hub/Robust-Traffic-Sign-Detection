"""
Phase 3 — Architecture Design & Validation Script.

MICAI 2025 | Robust Traffic Sign Detection

This script:
  1. Registers custom modules (SubPixelConv, CoordAtt) into the Ultralytics
     module namespace so that parse_model() can resolve them from the YAML.
  2. Monkey-patches parse_model() so that our custom modules are treated as
     base_modules (auto-injecting c1/c2 with width-multiplier scaling).
  3. Builds the model from models/configs/yolov8_custom.yaml.
  4. Runs a dry-run forward pass with a (1, 3, 640, 640) tensor.
  5. Inspects the architecture and validates module counts + parameters.
  6. Saves a validation report to results/phase3_architecture_validation.txt.

Usage:
    python scripts/phase3_architecture_design.py

The `register_custom_modules()` function is exposed for external import
(e.g., by train.py) so that registry injection can happen before YOLO
instantiation in any script.
"""

from __future__ import annotations

import sys
import os
import inspect
import textwrap

# ── Ensure project root is on sys.path ──────────────────────────
# This script lives in scripts/, so project root is one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═════════════════════════════════════════════════════════════════
# PUBLIC API — Reusable registration function (importable by train.py)
# ═════════════════════════════════════════════════════════════════

def register_custom_modules() -> None:
    """Register SubPixelConv and CoordAtt into the Ultralytics module namespace.

    Must be called BEFORE instantiating any YOLO model with the custom YAML.
    Idempotent — safe to call multiple times.

    This function:
      1. Injects module classes into ``ultralytics.nn.modules`` and
         ``ultralytics.nn.tasks`` globals so that ``parse_model()``'s
         ``globals()[m]`` name resolution finds them.
      2. Monkey-patches ``parse_model()`` by modifying its source code to
         include our modules in the ``base_modules`` frozenset, then
         exec-ing the modified source within the tasks module's globals.
    """
    from models.custom_modules import SubPixelConv, CoordAtt
    import ultralytics.nn.modules as ultralytics_modules
    import ultralytics.nn.tasks as ultralytics_tasks

    # ── Step 1: Inject into namespace for name resolution ───────
    setattr(ultralytics_modules, 'SubPixelConv', SubPixelConv)
    setattr(ultralytics_modules, 'CoordAtt', CoordAtt)
    ultralytics_tasks.SubPixelConv = SubPixelConv
    ultralytics_tasks.CoordAtt = CoordAtt

    # ── Step 2: Monkey-patch parse_model ────────────────────────
    _patch_parse_model(ultralytics_tasks, SubPixelConv, CoordAtt)


def _patch_parse_model(tasks_module, SubPixelConv, CoordAtt) -> None:
    """Replace ``parse_model`` with a source-modified version.

    We get the source code of the original ``parse_model``, add our custom
    modules to the ``base_modules`` frozenset definition, and exec the
    modified source in the tasks module's globals namespace. This ensures:
      - ``globals()[m]`` resolves correctly (same namespace as original)
      - Our modules get the same ``(c1, c2, *args)`` injection as built-ins
      - Channel tracking (``c2``) works correctly for all layers
    """
    original_fn = tasks_module.parse_model

    # Guard: don't double-patch
    if getattr(original_fn, '_custom_patched', False):
        return

    # Get the source code of the original parse_model
    source = inspect.getsource(original_fn)

    # Dedent to remove any leading whitespace
    source = textwrap.dedent(source)

    # ── Modify the source to add our modules to base_modules ───
    # We insert SubPixelConv and CoordAtt into the base_modules frozenset.
    # The frozenset definition in the source looks like:
    #   base_modules = frozenset(
    #       {
    #           Classify,
    #           Conv,
    #           ...
    #           A2C2f,
    #       }
    #   )
    # We add our modules right after A2C2f (the last entry).

    # Find the insertion point: right before the closing "}" of base_modules
    insertion_marker = "A2C2f,"
    if insertion_marker not in source:
        # Try alternative markers if the source format differs
        insertion_marker = "A2C2f,\n"
        if insertion_marker not in source:
            raise RuntimeError(
                "Could not find 'A2C2f,' in parse_model source to insert "
                "custom modules. The ultralytics version may be incompatible."
            )

    source = source.replace(
        insertion_marker,
        insertion_marker + "\n            SubPixelConv,\n            CoordAtt,",
        1,  # Only replace first occurrence (in base_modules, not repeat_modules)
    )

    # ── Compile and exec in the tasks module's globals ──────────
    # This ensures globals()[m] resolves from the tasks module namespace.
    exec_globals = tasks_module.__dict__
    exec(compile(source, "<patched_parse_model>", "exec"), exec_globals)

    # The exec created a new 'parse_model' in exec_globals
    patched_fn = exec_globals['parse_model']
    patched_fn._custom_patched = True

    # Replace the module-level function
    tasks_module.parse_model = patched_fn


# ═════════════════════════════════════════════════════════════════
# MAIN VALIDATION PIPELINE
# ═════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the full Phase 3 architecture validation pipeline."""

    # ── PART 1: Registry Injection ──────────────────────────────
    # CRITICAL: This MUST happen BEFORE `from ultralytics import YOLO`.
    register_custom_modules()

    print("[PHASE 3] ✓ Custom modules injected into Ultralytics registry.")

    # ── PART 2: Model Construction ──────────────────────────────
    from ultralytics import YOLO
    import torch

    # Build model from our custom YAML (this triggers parse_model internally)
    YAML_PATH = os.path.join(PROJECT_ROOT, 'models', 'configs', 'yolov8_custom.yaml')
    model = YOLO(YAML_PATH)
    print("[PHASE 3] ✓ Model built successfully from custom YAML.")

    # ── PART 3: Dry-Run Forward Pass Validation ─────────────────
    # Dummy input: batch=1, channels=3 (RGB), height=640, width=640
    # This is the standard YOLOv8 training resolution.
    dummy_input = torch.randn(1, 3, 640, 640)

    # Set model to eval mode to avoid BatchNorm issues with batch_size=1
    model.model.eval()

    with torch.no_grad():
        output = model.model(dummy_input)

    print("[PHASE 3] ✓ Forward pass completed — no shape mismatch errors.")

    # ── PART 4: Architecture Inspection ─────────────────────────
    # 1. Print full model summary
    model.model.info(verbose=True)

    # 2. Verify detection head channel configuration
    detect_layer = model.model.model[-1]  # Last layer is Detect
    print(f"\n[PHASE 3] Detection head info:")
    print(f"  Detect layer type: {type(detect_layer).__name__}")
    print(f"  Number of classes (nc): {detect_layer.nc}")
    print(f"  Number of detection layers: {detect_layer.nl}")

    # 3. Count total parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"\n[PHASE 3] Parameter count:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 4. Verify custom modules are present in the model
    custom_module_count = {
        'SubPixelConv': 0,
        'CoordAtt': 0,
    }
    for name, module in model.model.named_modules():
        cls_name = type(module).__name__
        if cls_name in custom_module_count:
            custom_module_count[cls_name] += 1
            print(f"  Found {cls_name} at: {name}")

    print(f"\n[PHASE 3] Custom module census:")
    for mod_name, count in custom_module_count.items():
        expected = 2 if mod_name == 'SubPixelConv' else 4
        status = '✓' if count == expected else f'✗ EXPECTED {expected}'
        print(f"  {mod_name}: {count} instances {status}")

    assert custom_module_count['SubPixelConv'] == 2, \
        f"Expected 2 SubPixelConv, found {custom_module_count['SubPixelConv']}"
    assert custom_module_count['CoordAtt'] == 4, \
        f"Expected 4 CoordAtt, found {custom_module_count['CoordAtt']}"

    # ── PART 5: Save Validation Report ──────────────────────────
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'phase3_architecture_validation.txt')

    with open(report_path, 'w') as f:
        f.write("Phase 3 — Architecture Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"YAML:       models/configs/yolov8_custom.yaml\n")
        f.write(f"Classes:    {detect_layer.nc}\n")
        f.write(f"Det layers: {detect_layer.nl}\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Trainable:  {trainable_params:,}\n\n")
        f.write("Custom Modules:\n")
        for mod_name, count in custom_module_count.items():
            f.write(f"  {mod_name}: {count}\n")
        f.write(f"\nDry-run input shape:  (1, 3, 640, 640)\n")
        f.write(f"Forward pass: SUCCESS\n")
        f.write(f"\nReport generated by: scripts/phase3_architecture_design.py\n")

    print(f"\n[PHASE 3] ✓ Validation report saved to: {report_path}")

    # ── PART 7: Final success banner ────────────────────────────
    print("\n" + "=" * 60)
    print("[PHASE 3 COMPLETE] Architecture validated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
