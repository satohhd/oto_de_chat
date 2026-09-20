"""チャットのようにテキストを入力すると、音階に変換してスピーカーから再生する。

使い方:
    python3 chat_speak.py [1音の長さ(秒)] [--phone]

例:
    python3 chat_speak.py                # 通常モード、250ms
    python3 chat_speak.py 0.1 --phone    # 電話回線用の周波数、100ms

入力 "exit" または空入力(Ctrl+D)で終了。
"""
import subprocess
import sys

from encode import SAMPLE_RATE, SYMBOL_DURATION, FADE_MS, FREQS, PHONE_FREQS, NOTE_NAMES, encode_to_wav, text_to_symbols

OUT_PATH = "chat_out.wav"


def play(path: str) -> None:
    subprocess.run(["aplay", "-q", path], check=True)


def main() -> None:
    args = sys.argv[1:]
    phone_mode = "--phone" in args
    if phone_mode:
        args.remove("--phone")

    duration = float(args[0]) if len(args) > 0 else SYMBOL_DURATION
    freqs = PHONE_FREQS if phone_mode else FREQS
    fade_ms = min(FADE_MS, duration * 1000 * 0.1)

    mode_label = "電話回線用(C5-B5)" if phone_mode else "通常(C5-B5)"
    print(f"[音声チャット] モード: {mode_label}  1音: {duration * 1000:.0f}ms")
    print("テキストを入力してEnterで再生します。終了は exit か Ctrl+D。")

    while True:
        try:
            text = input("> ")
        except EOFError:
            print()
            break

        if not text or text.strip().lower() == "exit":
            break

        symbols = text_to_symbols(text)
        notes = " ".join(NOTE_NAMES[s] for s in symbols)
        seconds = len(symbols) * duration
        print(f"  音階列: {notes}  ({len(symbols)}音 / {seconds:.2f}秒)")

        encode_to_wav(text, OUT_PATH, duration, fade_ms, freqs)
        play(OUT_PATH)


if __name__ == "__main__":
    main()
