"""Pinned Light-ASD inference source; model weights live outside this package."""

from pathlib import Path

UPSTREAM_REVISION = "00a4a4efcd0cef91deb3154edd5dd940b65a15bd"
SOURCE_ROOT = Path(__file__).resolve().parent


def load_asd():
    """Import the complete inference closure with an actionable packaging error."""
    required = (
        "ASD.py", "loss.py", "model/Model.py", "model/Encoder.py",
        "model/Classifier.py",
    )
    missing = [name for name in required if not (SOURCE_ROOT / name).is_file()]
    if missing:
        raise ImportError(
            f"Incomplete vendored Light-ASD {UPSTREAM_REVISION} at {SOURCE_ROOT}: "
            f"missing {', '.join(missing)}. Rebuild the inference_ray package/image."
        )
    from .ASD import ASD

    return ASD
