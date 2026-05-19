
#!/usr/bin/env python3
# 检查 GPU / CUDA / PyTorch 可用性的小脚本

import subprocess
import shutil
import sys

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        return out.strip()
    except Exception as e:
        return None

def main():
    import torch
    print("PyTorch:", torch.__version__)
    print("PyTorch built with CUDA:", torch.version.cuda)
    try:
        print("cuDNN version:", torch.backends.cudnn.version())
    except Exception:
        pass

    # nvidia-smi
    if shutil.which("nvidia-smi"):
        print("\nnvidia-smi:")
        print(run_cmd("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits"))
    else:
        print("\nnvidia-smi: not found")

    # nvcc
    if shutil.which("nvcc"):
        print("\nnvcc --version:")
        print(run_cmd("nvcc --version"))
    else:
        print("\nnvcc: not found")

    print("\nTorch CUDA available:", torch.cuda.is_available())
    cnt = torch.cuda.device_count()
    print("CUDA device count:", cnt)

    for i in range(cnt):
        prop = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"\nDevice {i}: {prop.name}")
        print(f"  total_memory (MB): {prop.total_memory // (1024**2)}")
        print(f"  compute capability: {cap[0]}.{cap[1]}")
        print(f"  multi_processor_count: {prop.multi_processor_count}")
        # quick test: allocate and do a small op on the device
        try:
            t = torch.ones(2, 2, device=f"cuda:{i}")
            t = t * 2
            print("  simple CUDA op: OK")
        except Exception as e:
            print("  simple CUDA op: FAILED ->", e)

    # extra: try to detect mismatch warning (best-effort)
    # capture current stderr stdout of a dummy cuda import to see warnings
    print("\nQuick compatibility hint:")
    try:
        # try to allocate a small tensor on default device to see runtime errors
        if torch.cuda.is_available():
            torch.tensor([1.0], device="cuda")
            print("  Allocation on default CUDA device: OK")
        else:
            print("  CUDA not available in PyTorch")
    except Exception as e:
        print("  Allocation on default CUDA device failed:", e)

    return 0

if __name__ == '__main__':
    sys.exit(main())