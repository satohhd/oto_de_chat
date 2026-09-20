"""マイクで録音 → 無音トリミング → デコードしてテキスト化 → 確認用に再エンコード、までの一連の処理。

使い方:
    python3 record_and_decode.py [録音秒数] [1音の長さ(秒)] [--phone]

例:
    python3 record_and_decode.py 10 0.25          # 通常モードで10秒録音
    python3 record_and_decode.py 10 0.25 --phone  # 電話回線用の周波数で復号
"""
import subprocess
import sys

import numpy as np

from encode import SAMPLE_RATE, SYMBOL_DURATION, FADE_MS, FREQS, PHONE_FREQS, encode_to_wav, symbols_to_text
from decode import decode_audio, read_wav, find_sync_start


def record(path: str, seconds: float) -> None:
    # arecordの-dは整数秒のみ受け付けるため、端数分は切り上げて余分に録音する
    duration_arg = max(1, int(np.ceil(seconds)))
    subprocess.run(
        ["arecord", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-d", str(duration_arg), path],
        check=True,
    )


def trim_active_range(audio: np.ndarray, block_size: int, threshold_ratio: float = 0.3) -> np.ndarray:
    """録音の前後にある無音区間を除き、信号が乗っているシンボル区間だけを残す。"""
    n_blocks = len(audio) // block_size
    if n_blocks == 0:
        return audio[:0]

    energies = np.array([
        np.sqrt(np.mean(audio[i * block_size:(i + 1) * block_size] ** 2))
        for i in range(n_blocks)
    ])
    peak = energies.max()
    if peak == 0:
        return audio[:0]

    active = np.where(energies >= peak * threshold_ratio)[0]
    if len(active) == 0:
        return audio[:0]

    start, end = active[0], active[-1]
    return audio[start * block_size:(end + 1) * block_size]


def main() -> None:
    args = sys.argv[1:]
    phone_mode = "--phone" in args
    if phone_mode:
        args.remove("--phone")

    seconds = float(args[0]) if len(args) > 0 else 10.0
    duration = float(args[1]) if len(args) > 1 else SYMBOL_DURATION
    freqs = PHONE_FREQS if phone_mode else FREQS
    fade_ms = min(FADE_MS, duration * 1000 * 0.1)

    recorded_path = "recorded_mic.wav"
    print(f"[{seconds:.1f}秒間 録音します] 音を鳴らしてマイクに聞かせてください...")
    record(recorded_path, seconds)
    print(f"録音完了: {recorded_path}")

    audio = read_wav(recorded_path)
    block_size = int(SAMPLE_RATE * duration)
    trimmed = trim_active_range(audio, block_size)
    print(f"無音トリミング後: {len(trimmed) / SAMPLE_RATE:.2f}秒 "
          f"({len(trimmed) // block_size}シンボル)")

    # トリミングだけではブロック境界とズレる可能性があるため、同期音を探して正確な開始位置に合わせる
    result = find_sync_start(trimmed, duration, freqs, search_samples=block_size * 2)
    if result is None:
        print("(同期音が見つからなかったため、トリミング位置からそのままデコードします)")
        start, ratio = 0, 1.0
    else:
        start, ratio = result
        print(f"同期音を検出、データ開始位置に補正しました(ドリフト比: {ratio:.4f})")

    symbols = decode_audio(trimmed[start:], duration, freqs, ratio)
    text = symbols_to_text(symbols)
    print(f"復元したメッセージ: {text!r}")

    verify_path = "recorded_reencoded.wav"
    encode_to_wav(text, verify_path, duration, fade_ms, freqs)
    print(f"確認用に再エンコード: {verify_path}")


if __name__ == "__main__":
    main()
