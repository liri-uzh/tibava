from inference_ray.plugin import AnalyserPlugin, AnalyserPluginManager  # type: ignore
from tibava_data import AudioData, AnnotationData, ListData, Annotation  # type: ignore

from tibava_data import DataManager, Data  # type: ignore

from typing import Callable, Dict
import logging

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
}

default_parameters = {"language_code": None, "require_alignment": False}

requires = {
    "audio": AudioData,
}

provides = {
    "annotations": ListData,
}


def get_speaker_turns(
    speaker_segments, gap: float = 0.01
) -> Dict[str, list[Annotation]]:
    speaker_turns = []
    for segment in sorted(speaker_segments, key=lambda x: x["start"]):
        spk_turn_segment = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment.get("text", "").strip(),
            "speaker": segment.get("speaker", "Unknown"),
        }

        if not speaker_turns:
            speaker_turns.append(spk_turn_segment)
        else:
            last_turn = speaker_turns[-1]
            if last_turn["speaker"] == spk_turn_segment["speaker"]:
                last_turn["end"] = spk_turn_segment["end"]
                last_turn["text"] += " " + spk_turn_segment["text"]
            else:
                if spk_turn_segment["start"] - last_turn["end"] <= gap:
                    spk_turn_segment["start"] = last_turn["end"] + gap
                speaker_turns.append(spk_turn_segment)

    speakers = {}
    for turn in speaker_turns:
        if not turn["speaker"] in speakers.keys():
            speakers[turn["speaker"]] = []
        speakers[turn["speaker"]].append(
            Annotation(start=turn["start"], end=turn["end"], labels=[turn["text"]])
        )
    return speakers


@AnalyserPluginManager.export("whisper_x")
class WhisperX(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.1",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # inference_config = self.config.get("inference", None)
        # self.server = InferenceServer.build(inference_config.get("type"), inference_config.get("params", {}))

        self.model = None
        self.diarize_model = None
        self.alignment_model = None
        self.model_name = self.config.get("model", "whisper_x")

    def __call__(self, inputs, data_manager, parameters=None, callbacks=None):
        # The base plugin retains parameters between calls. Do not retain a
        # previous audio workflow's alignment requirement for a transcript job.
        parameters = {"require_alignment": False, **(parameters or {})}
        return super().__call__(inputs, data_manager, parameters, callbacks=callbacks)

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        import librosa
        import torch
        import whisperx

        language_code = (parameters or {}).get("language_code") or None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is None:
            self.model = whisperx.load_model(
                "large-v3",
                device=device,
                compute_type="int8",
                language=language_code,
            )  # TODO originally compute_type="float16" but not supported by m1. probably change back for production
            self.diarize_model = whisperx.diarize.DiarizationPipeline(device=device)
            self.device = device

        with (
            inputs["audio"] as input_data,
            data_manager.create_data("ListData") as output_data,
        ):
            with input_data.open_audio("r") as f_audio:
                y, sr = librosa.load(f_audio, sr=16000)
                transcription = self.model.transcribe(
                    audio=y, batch_size=8, language=language_code
                )

                alignment_language = language_code or transcription["language"]
                # always instantiate new alignment model to match current language
                try:
                    self.alignment_model, self.metadata = whisperx.load_align_model(
                        language_code=alignment_language, device=device
                    )
                except ValueError as exc:
                    if str(exc) != (
                        f"No default align-model for language: {alignment_language}"
                    ):
                        raise
                    if (parameters or {}).get("require_alignment", False):
                        raise ValueError(
                            "WhisperX alignment is required for this workflow, but "
                            "no default alignment model is available for language "
                            f"'{alignment_language}'. Select the correct language "
                            "explicitly if detection was wrong."
                        ) from exc
                    logging.warning(
                        "No default WhisperX alignment model for language '%s'; "
                        "continuing without alignment. Using coarse transcription "
                        "timestamps; speaker attribution may be less accurate.",
                        alignment_language,
                    )
                    aligned_transcription = transcription
                else:
                    aligned_transcription = whisperx.align(
                        transcription["segments"],
                        self.alignment_model,
                        self.metadata,
                        y,
                        device,
                        return_char_alignments=False,
                    )

                diarize_segments = self.diarize_model(y)
                speaker_transcription = whisperx.assign_word_speakers(
                    diarize_segments, aligned_transcription
                )
                speaker_turns = get_speaker_turns(speaker_transcription["segments"])

                for speaker, annotations in speaker_turns.items():
                    with output_data.create_data("AnnotationData") as ann_data:
                        ann_data.annotations.extend(annotations)
                        ann_data.name = speaker

                self.update_callbacks(callbacks, progress=1.0)
                return {"annotations": output_data}
