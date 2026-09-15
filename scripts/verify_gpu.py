"""Fail unless PyTorch, ONNX Runtime, and Ray can use the assigned GPU."""

import torch
import onnxruntime as ort
import ray


print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA")

print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("VRAM bytes:", torch.cuda.get_device_properties(0).total_memory)

print("ONNX Runtime:", ort.__version__)
providers = ort.get_available_providers()
print("ONNX Runtime providers:", providers)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("ERROR: ONNX Runtime CUDAExecutionProvider is unavailable")

ray.init(address="auto")
resources = ray.cluster_resources()
print("Ray cluster resources:", resources)
if resources.get("GPU", 0) < 1:
    raise SystemExit("ERROR: Ray does not report one GPU resource")
