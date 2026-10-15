"""Vendored imports and construction boundaries; no GPU, weights, or inference."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

# Also runnable directly from a source checkout, without workspace installation.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from inference_ray.vendor import light_asd


def load_plugin():
    infrastructure = ModuleType("inference_ray.plugin")

    class Plugin:
        def __init_subclass__(cls, **kwargs):
            pass

        def __init__(self, *args, **kwargs):
            pass

    infrastructure.AnalyserPlugin = Plugin
    infrastructure.AnalyserPluginManager = SimpleNamespace(
        export=lambda name: lambda cls: cls
    )
    data = ModuleType("tibava_data")
    for name in ("AnnotationData", "Annotation", "AudioData", "FacesData",
                 "VideoData", "DataManager", "Data"):
        setattr(data, name, type(name, (), {}))
    utils = ModuleType("tibava_utils")
    utils.VideoDecoder = object
    path = Path(__file__).parents[1] / "src/inference_ray/plugins/active_speaker_detection.py"
    spec = importlib.util.spec_from_file_location("active_speaker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "inference_ray.plugin": infrastructure,
        "tibava_data": data,
        "tibava_utils": utils,
    }):
        spec.loader.exec_module(module)
    return module


class ActiveSpeakerImportTests(unittest.TestCase):
    def test_real_vendored_import_closure(self):
        # Imports are real, including Torch, NumPy, pandas and tqdm.
        before = sys.path.copy()
        ASD = light_asd.load_asd()
        root = light_asd.SOURCE_ROOT
        for name, relative in {
            "ASD": "ASD.py", "loss": "loss.py",
            "model.Model": "model/Model.py",
            "model.Encoder": "model/Encoder.py",
            "model.Classifier": "model/Classifier.py",
        }.items():
            module = importlib.import_module("inference_ray.vendor.light_asd." + name)
            self.assertEqual(Path(module.__file__).resolve(), root / relative)
        self.assertEqual(ASD.__module__, "inference_ray.vendor.light_asd.ASD")
        self.assertEqual(light_asd.UPSTREAM_REVISION,
                         "00a4a4efcd0cef91deb3154edd5dd940b65a15bd")
        self.assertEqual(sys.path, before)
        for name in ("loadParameters", "evaluate_network", "to", "eval"):
            self.assertTrue(callable(getattr(ASD, name)))
        model = importlib.import_module("inference_ray.vendor.light_asd.model.Model")
        for name in ("forward_audio_frontend", "forward_visual_frontend",
                     "forward_audio_visual_backend"):
            self.assertTrue(callable(getattr(model.ASD_Model, name)))

    def test_missing_source_has_clear_error(self):
        actual = Path.is_file
        for missing in ("ASD.py", "loss.py", "model/Encoder.py"):
            with self.subTest(missing=missing), patch.object(
                Path, "is_file",
                lambda p: False if p == light_asd.SOURCE_ROOT / missing else actual(p),
            ):
                with self.assertRaisesRegex(ImportError, "Incomplete vendored Light-ASD") as error:
                    light_asd.load_asd()
                self.assertIn(missing, str(error.exception))
                self.assertIn("Rebuild", str(error.exception))

    def test_real_construction_and_state_dict_loading_with_cuda_mocked(self):
        import torch
        ASD = light_asd.load_asd()
        with patch.object(torch.nn.Module, "cuda", autospec=True,
                          side_effect=lambda module: module) as cuda:
            model = ASD()
        self.assertEqual(cuda.call_count, 3)
        key, tensor = next(iter(model.state_dict().items()))
        replacement = torch.ones_like(tensor)
        path = Path("/not-read/finetuning_TalkSet.model")
        with patch.object(torch, "load", return_value={key: replacement}) as load:
            model.loadParameters(path)
        load.assert_called_once_with(path)
        self.assertTrue(torch.equal(model.state_dict()[key], replacement))
        # This is construction + a mocked state dict only, never inference.

    def test_plugin_uses_vendored_loader_and_preserves_checkpoint_paths(self):
        import torch
        plugin = load_plugin()
        dependencies = {name: ModuleType(name) for name in
                        ("librosa", "python_speech_features", "cv2")}
        for selector, filename in (("talkset", "finetuning_TalkSet.model"),
                                   ("ava", "pretrain_AVA_CVPR.model")):
            model = MagicMock()
            constructor = MagicMock(return_value=model)
            with self.subTest(model=selector), patch.dict(sys.modules, dependencies), \
                 patch.object(torch.cuda, "is_available", return_value=True), \
                 patch.object(light_asd, "load_asd", return_value=constructor) as loader, \
                 patch.dict(plugin.default_config, {"model": selector}):
                # Stop at missing request inputs, immediately after model setup.
                with self.assertRaises(KeyError):
                    plugin.ActiveSpeakerDetection().call({}, None)
                loader.assert_called_once_with()
                constructor.assert_called_once_with()
                model.loadParameters.assert_called_once_with(
                    Path("/models/asd/Light-ASD/weight") / filename)
                model.to.assert_called_once_with("cuda")
                model.eval.assert_called_once_with()

    def test_cpu_actor_fails_before_cuda_construction_or_loading(self):
        import torch
        plugin = load_plugin()
        dependencies = {name: ModuleType(name) for name in
                        ("librosa", "python_speech_features", "cv2")}
        constructor = MagicMock()
        with patch.dict(sys.modules, dependencies), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(light_asd, "load_asd", return_value=constructor):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA.*Ray GPU reservation"):
                plugin.ActiveSpeakerDetection().call({}, None)
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
