"""Language sets from docs/whisperx-alignment-audit.md.

Production WhisperX commit 771b4a14a9486f8fd5aef18ef49e35d639523dd3.
Tests keep these scoped backend definitions in sync with the frontend and audit.
"""

WHISPER_LANGUAGES = frozenset("""
af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi
fo fr gl gu ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb
ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru
sa sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi
yi yo yue zh
""".split())

WHISPERX_ALIGNMENT_LANGUAGES = frozenset("""
ar ca cs da de el en es eu fa fi fr gl he hi hr hu id it ja ka ko lv ml
nl nn no pl pt ro ru sk sl sv te tl tr uk ur vi zh
""".split())
