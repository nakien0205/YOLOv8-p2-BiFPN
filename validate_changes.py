#!/usr/bin/env python3
"""Validate Ultralytics changes: compile and model load."""

import sys
import py_compile
import traceback

print("=" * 60)
print("VALIDATION: Ultralytics Changes")
print("=" * 60)

# 1. Compile check
files_to_check = [
    'ultralytics/nn/modules/conv.py',
    'ultralytics/nn/tasks.py',
    'ultralytics/utils/loss.py'
]

print("\n[1] Compile Check...")
compile_ok = True
for fpath in files_to_check:
    try:
        py_compile.compile(fpath, doraise=True)
        print(f"  ✓ {fpath}")
    except py_compile.PyCompileError as e:
        print(f"  ✗ {fpath}")
        print(traceback.format_exc())
        compile_ok = False

if not compile_ok:
    print("\n❌ FAILED: Compilation errors detected")
    sys.exit(1)

print("\n[2] Model Load Test...")
try:
    from ultralytics import YOLO
    m = YOLO('ultralytics/cfg/models/v8/yolov8-p2-BiFPN.yaml')
    print(f"  ✓ Model loaded: {m.model is not None}")
    bbox_loss = m.model.args.get('bbox_loss', 'missing')
    print(f"  ✓ bbox_loss: {bbox_loss}")
except Exception as e:
    print(f"  ✗ Model load failed")
    print(traceback.format_exc())
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ SUCCESS: All validations passed")
print("=" * 60)
