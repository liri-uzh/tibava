# Light-ASD inference source

Upstream: https://github.com/Junhua-Liao/Light-ASD

Revision: `00a4a4efcd0cef91deb3154edd5dd940b65a15bd` (2023-10-31).
The original MIT LICENSE is included here and shipped as package data.

Vendored files and upstream Git blob IDs:

| File | Git blob |
| --- | --- |
| ASD.py | 2b7dbf13847332afc2ed2a746617c33828f6790a |
| loss.py | a7ffcc66ef24b4713f8b018df27055c5e1760f44 |
| model/Model.py | 1f3ab065c2119e7f49503149e34725c316f902ae |
| model/Encoder.py | 7815a655770edd7457036312c7a5d777321ca5f4 |
| model/Classifier.py | 73c0c52fe9f9c7b09216800ad3bec4c4ae785f1e |
| LICENSE | b755b4bbeda0b617c4431ec00942c7de8342362b |

Local changes: Python line endings/trailing whitespace normalized; sibling
imports in ASD.py and model/Model.py changed to relative imports. Package
initializers and the checked `load_asd()` entry point are TiBAV additions.
Model definitions, checkpoint loading, training/evaluation methods, and scoring
are unchanged. Training/evaluation data and helper scripts are not included;
their entry points are not supported by this inference-only distribution.

The source is installed inside `inference_ray.vendor.light_asd`, outside the
mounted `/models` directory. No checkout or network access is required at actor
startup to obtain Light-ASD source. Ray may still install runtime pip dependencies.
Weights remain external at `/models/asd/Light-ASD/weight/`; none are included.

Upstream `ASD()` uses CUDA unconditionally for the model and both losses. The
CUDA deployment reserves 0.1 GPU for visibility/scheduling, not a VRAM cap.
A CPU adaptation would replace those three constructor `.cuda()` calls with
device-aware transfers and add `map_location` in `loadParameters`; TiBAV's
inference calls already transfer input tensors to the chosen device. That
adaptation is not applied: the intended production path uses the L4 GPU.
