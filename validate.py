import os
import sys
os.chdir(r'D:\Python\Projects\ultralytics')

import py_compile

print("=== COMPILE CHECK ===")
files = [
    'ultralytics/nn/modules/conv.py',
    'ultralytics/nn/tasks.py',
    'ultralytics/utils/loss.py'
]

compile_success = True
for file in files:
    try:
        py_compile.compile(file, doraise=True)
        print(f"✓ {file}")
    except py_compile.PyCompileError as e:
        print(f"✗ {file}: {e}")
        compile_success = False

if compile_success:
    print("\n=== MODEL LOAD TEST ===")
    try:
        from ultralytics import YOLO
        m = YOLO('ultralytics/cfg/models/v8/yolov8-p2-BiFPN.yaml')
        loaded = m.model is not None
        bbox_loss = m.model.args.get('bbox_loss', 'missing')
        print(f"loaded: {loaded}")
        print(f"bbox_loss_default: {bbox_loss}")
        print("\n✓ Success: All files compile, model loads, bbox_loss configured")
    except Exception as e:
        print(f"✗ Model load failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n✗ Compilation failed, skipping model load test")
