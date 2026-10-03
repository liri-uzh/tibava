import definitions from "./whisperx-languages.json";

// Both sets match the production revision recorded in the alignment audit.
export const WHISPER_LANGUAGES = definitions.transcription;
export const WHISPERX_ALIGNMENT_LANGUAGES = WHISPER_LANGUAGES.filter(language =>
  definitions.alignmentCodes.includes(language.value)
);

export function whisperLanguageOptions(autoDetectLabel, requireAlignment = false) {
  return [
    { text: autoDetectLabel, value: "none" },
    ...(requireAlignment ? WHISPERX_ALIGNMENT_LANGUAGES : WHISPER_LANGUAGES),
  ];
}
