"""Exercise the plugin pipeline without model downloads or inference dependencies."""

import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


@dataclass
class Annotation:
    start: float
    end: float
    labels: list


class PluginStub:
    def __init_subclass__(cls, **kwargs):
        cls._parameters = kwargs.get("parameters", {}).copy()

    def __init__(self, config=None, **kwargs):
        self.config = config or {}

    def update_callbacks(self, callbacks, **kwargs):
        for callback in callbacks or []:
            callback.update(**kwargs)

    def __call__(self, inputs, data_manager, parameters=None, callbacks=None):
        # Match the real base plugin's retained-parameter behavior.
        self._parameters.update(parameters or {})
        return self.call(inputs, data_manager, self._parameters, callbacks=callbacks)


def load_plugin():
    # Isolate infrastructure imports; execute the actual plugin implementation.
    plugin_module = ModuleType("inference_ray.plugin")
    plugin_module.AnalyserPlugin = PluginStub
    plugin_module.AnalyserPluginManager = SimpleNamespace(
        export=lambda name: lambda cls: cls
    )
    data_module = ModuleType("tibava_data")
    for name in ("AudioData", "AnnotationData", "ListData", "DataManager", "Data"):
        setattr(data_module, name, type(name, (), {}))
    data_module.Annotation = Annotation
    path = Path(__file__).parents[1] / "src/inference_ray/plugins/whisper_x.py"
    spec = importlib.util.spec_from_file_location("whisper_x_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {
        "inference_ray.plugin": plugin_module,
        "tibava_data": data_module,
    }):
        spec.loader.exec_module(module)
    return module.WhisperX


class WhisperXTests(unittest.TestCase):
    def setUp(self):
        self.plugin = load_plugin()()
        self.audio = object()
        self.transcription = {
            "language": "cy",
            "segments": [{"start": 1.0, "end": 3.0, "text": " Hello "}],
        }
        self.aligned = {
            "segments": [{
                "start": 1.2, "end": 2.8, "text": " Hello ",
                "words": [{"start": 1.2, "end": 2.8, "word": "Hello"}],
            }],
        }
        self.model = Mock()
        self.model.transcribe.return_value = self.transcription
        self.diarizer = Mock(return_value=object())
        self.alignment_model = object()
        self.metadata = {"language": "en"}
        self.whisperx = ModuleType("whisperx")
        self.whisperx.load_model = Mock(return_value=self.model)
        self.whisperx.diarize = SimpleNamespace(
            DiarizationPipeline=Mock(return_value=self.diarizer)
        )
        self.whisperx.load_align_model = Mock(
            return_value=(self.alignment_model, self.metadata)
        )
        self.whisperx.align = Mock(return_value=self.aligned)

        def assign_speakers(diarization, transcript):
            # WhisperX supports segment-level speaker assignment without words.
            for segment in transcript["segments"]:
                segment["speaker"] = "SPEAKER_00"
            return transcript

        self.whisperx.assign_word_speakers = Mock(side_effect=assign_speakers)
        librosa = ModuleType("librosa")
        librosa.load = Mock(return_value=(self.audio, 16000))
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=Mock(return_value=False))
        imports = patch.dict("sys.modules", {
            "whisperx": self.whisperx, "librosa": librosa, "torch": torch,
        })
        imports.start()
        self.addCleanup(imports.stop)

        self.input_data = MagicMock()
        self.input_data.__enter__.return_value = self.input_data
        self.input_data.open_audio.return_value = nullcontext(object())
        self.annotation_data = SimpleNamespace(annotations=[], name=None)
        self.output_data = MagicMock()
        self.output_data.__enter__.return_value = self.output_data
        self.output_data.create_data.return_value = nullcontext(self.annotation_data)
        self.manager = Mock()
        self.manager.create_data.return_value = self.output_data
        self.callback = Mock()

    def call(self, language_code=None, require_alignment=False):
        return self.plugin.call(
            {"audio": self.input_data}, self.manager,
            parameters={"language_code": language_code, "require_alignment": require_alignment},
            callbacks=[self.callback],
        )

    def assert_output(self, result, start, end):
        self.assertEqual(result, {"annotations": self.output_data})
        self.manager.create_data.assert_called_once_with("ListData")
        self.output_data.create_data.assert_called_once_with("AnnotationData")
        self.assertEqual(self.annotation_data.name, "SPEAKER_00")
        self.assertEqual(
            self.annotation_data.annotations,
            [Annotation(start=start, end=end, labels=["Hello"])],
        )
        self.diarizer.assert_called_once_with(self.audio)
        self.callback.update.assert_called_once_with(progress=1.0)

    def assert_alignment(self, language):
        self.whisperx.load_align_model.assert_called_once_with(
            language_code=language, device="cpu"
        )
        self.whisperx.align.assert_called_once_with(
            self.transcription["segments"], self.alignment_model, self.metadata,
            self.audio, "cpu", return_char_alignments=False,
        )
        self.whisperx.assign_word_speakers.assert_called_once_with(
            self.diarizer.return_value, self.aligned
        )

    def test_explicit_language_overrides_reported_language(self):
        result = self.call("en")
        self.whisperx.load_model.assert_called_once_with(
            "large-v3", device="cpu", compute_type="int8", language="en"
        )
        self.model.transcribe.assert_called_once_with(
            audio=self.audio, batch_size=8, language="en"
        )
        self.assert_alignment("en")
        self.assert_output(result, 1.2, 2.8)

    def test_unsupported_detection_continues_with_original_transcription(self):
        self.whisperx.load_align_model.side_effect = ValueError(
            "No default align-model for language: cy"
        )
        with self.assertLogs(level="WARNING") as logs:
            result = self.call()
        self.assertIn(
            "No default WhisperX alignment model for language 'cy'; "
            "continuing without alignment.", logs.output[0],
        )
        self.whisperx.load_align_model.assert_called_once_with(
            language_code="cy", device="cpu"
        )
        self.whisperx.align.assert_not_called()
        self.whisperx.assign_word_speakers.assert_called_once_with(
            self.diarizer.return_value, self.transcription
        )
        self.assertNotIn("words", self.transcription["segments"][0])
        self.assert_output(result, 1.0, 3.0)

    def test_supported_detection_still_aligns(self):
        self.transcription["language"] = "en"
        result = self.call()
        self.whisperx.load_model.assert_called_once_with(
            "large-v3", device="cpu", compute_type="int8", language=None
        )
        self.model.transcribe.assert_called_once_with(
            audio=self.audio, batch_size=8, language=None
        )
        self.assert_alignment("en")
        self.assert_output(result, 1.2, 2.8)

    def test_required_alignment_rejects_unsupported_language(self):
        self.whisperx.load_align_model.side_effect = ValueError(
            "No default align-model for language: cy"
        )
        with self.assertRaisesRegex(ValueError, "alignment is required.*language 'cy'"):
            self.call(require_alignment=True)
        self.whisperx.align.assert_not_called()
        self.diarizer.assert_not_called()
        self.whisperx.assign_word_speakers.assert_not_called()
        self.callback.update.assert_not_called()

    def test_required_alignment_accepts_supported_language(self):
        self.transcription["language"] = "en"
        result = self.call(require_alignment=True)
        self.assert_alignment("en")
        self.assert_output(result, 1.2, 2.8)

    def test_alignment_requirement_does_not_leak_into_next_request(self):
        self.whisperx.load_align_model.side_effect = ValueError(
            "No default align-model for language: cy"
        )
        inputs = {"audio": self.input_data}
        with self.assertRaises(ValueError):
            self.plugin(inputs, self.manager, {"require_alignment": True})
        self.manager.reset_mock()
        with self.assertLogs(level="WARNING"):
            result = self.plugin(inputs, self.manager, callbacks=[self.callback])
        self.assert_output(result, 1.0, 3.0)

    def test_wrappers_enforce_their_alignment_policy(self):
        helper_path = Path(__file__).parents[2] / "backend/tests/test_whisper_x_language.py"
        spec = importlib.util.spec_from_file_location("backend_test_helpers", helper_path)
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        self.whisperx.load_align_model.side_effect = ValueError(
            "No default align-model for language: cy"
        )
        for name in helpers.CALLERS:
            with self.subTest(plugin=name):
                module = helpers.load_task(name)
                task = getattr(module, helpers.CALLERS[name][1])()
                task.upload_video = Mock(return_value="video-id")
                stages = []

                def run_analyser(client, plugin, parameters=None, **kwargs):
                    stages.append(plugin)
                    if plugin == "video_to_audio":
                        return {"audio": "audio-id"}, {}
                    if plugin == "whisper_x":
                        self.plugin({"audio": self.input_data}, self.manager, parameters)
                        if name == "whisper_x":
                            raise helpers.StopAfterWhisperX
                        return {"annotations": "annotations-id"}, {}
                    raise helpers.StopAfterWhisperX

                task.run_analyser = Mock(side_effect=run_analyser)
                if name in ("audio_emotion", "audio_gender", "audio_classification"):
                    with self.assertRaisesRegex(ValueError, "alignment is required"):
                        task({"segment_type": "Speaker", "model_type": "Multilingual"})
                    self.assertEqual(stages, ["video_to_audio", "whisper_x"])
                else:
                    with self.assertLogs(level="WARNING"), self.assertRaises(helpers.StopAfterWhisperX):
                        task({"segment_type": "Speaker", "model_type": "Multilingual"})
                    expected = ["video_to_audio", "whisper_x"]
                    if name != "whisper_x":
                        expected.append(name)
                    self.assertEqual(stages, expected)

    def test_serializer_uses_segment_speakers_and_discards_word_metadata(self):
        speaker_turns = self.plugin.call.__globals__["get_speaker_turns"]([
            {
                "start": 0.0, "end": 10.0, "text": " First. Second. ",
                "speaker": "A", "avg_logprob": -0.1,
                "words": [{"word": "Second", "start": 6.0, "end": 9.0, "speaker": "B"}],
            },
        ])
        self.assertEqual(speaker_turns, {
            "A": [Annotation(start=0.0, end=10.0, labels=["First. Second."])],
        })

    def test_sentence_boundaries_can_change_serialized_speaker_turns(self):
        serialize = self.plugin.call.__globals__["get_speaker_turns"]
        # A merged VAD chunk receives its dominant speaker A. Aligned sentence
        # spans can instead overlap different speakers, producing different crops.
        raw = [{"start": 0.0, "end": 10.0, "text": "First. Second.", "speaker": "A"}]
        aligned = [
            {"start": 0.5, "end": 5.5, "text": "First.", "speaker": "A"},
            {"start": 6.5, "end": 9.5, "text": "Second.", "speaker": "B"},
        ]
        self.assertEqual(serialize(raw), {
            "A": [Annotation(start=0.0, end=10.0, labels=["First. Second."])],
        })
        self.assertEqual(serialize(aligned), {
            "A": [Annotation(start=0.5, end=5.5, labels=["First."])],
            "B": [Annotation(start=6.5, end=9.5, labels=["Second."])],
        })

    def test_empty_language_uses_detection(self):
        self.transcription["language"] = "en"
        self.call("")
        self.model.transcribe.assert_called_once_with(
            audio=self.audio, batch_size=8, language=None
        )
        self.assert_alignment("en")

    def test_explicit_language_is_used_with_cached_model(self):
        self.plugin.model = self.model
        self.plugin.diarize_model = self.diarizer
        self.call("en")
        self.whisperx.load_model.assert_not_called()
        self.model.transcribe.assert_called_once_with(
            audio=self.audio, batch_size=8, language="en"
        )
        self.assert_alignment("en")

    def test_unexpected_alignment_model_errors_propagate(self):
        for error in (ValueError("Model download failed"), RuntimeError("CUDA error")):
            with self.subTest(error=error):
                self.whisperx.load_align_model.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    self.call()
                self.assertIs(raised.exception, error)
        self.whisperx.align.assert_not_called()
        self.diarizer.assert_not_called()

    def test_alignment_errors_propagate(self):
        error = ValueError("Invalid alignment input")
        self.whisperx.align.side_effect = error
        with self.assertRaises(ValueError) as raised:
            self.call("en")
        self.assertIs(raised.exception, error)
        self.diarizer.assert_not_called()

    def test_transcription_errors_propagate(self):
        error = RuntimeError("Transcription failed")
        self.model.transcribe.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self.call()
        self.assertIs(raised.exception, error)
        self.whisperx.load_align_model.assert_not_called()

    def test_diarization_errors_propagate_after_fallback(self):
        self.whisperx.load_align_model.side_effect = ValueError(
            "No default align-model for language: cy"
        )
        error = RuntimeError("Diarization failed")
        self.diarizer.side_effect = error
        with self.assertLogs(level="WARNING"), self.assertRaises(RuntimeError) as raised:
            self.call()
        self.assertIs(raised.exception, error)
        self.callback.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
