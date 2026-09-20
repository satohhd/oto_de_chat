"""7音階（ドレミファソラシ）だけでデータを音にエンコードするサンプル。

方式:
- 7音 = 0〜6 の7進数1桁として扱う（和音なし、単音のみ）
- 7^3 = 343 >= 256 なので、1バイト = 7進数3桁 = 3音 で表現できる
- 1音 = 250ms（1秒を4分割）→ 4音/秒
"""
import wave
import numpy as np

SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.25  # seconds (1秒を4分割)
FADE_MS = 8  # クリックノイズ防止のフェード時間（SYMBOL_DURATIONの1割目安）

# ド レ ミ ファ ソ ラ シ (C5スタートの長音階。C4から1オクターブ上げてあり、
# 隣接音同士の周波数差(Hz)も比例して約2倍に広がるため、同じFFT分解能でも
# 判定マージンが大きくなり、誤認識に強くなる)
NOTE_NAMES = ["ド", "レ", "ミ", "ファ", "ソ", "ラ", "シ"]
FREQS = [523.26, 587.32, 659.26, 698.46, 784.00, 880.00, 987.76]

# 電話回線用（C5スタート）。電話の音声帯域(約300〜3400Hz)に収まるよう1オクターブ上げたもの。
# 通常版のド(261.63Hz)は300Hzを下回り帯域外フィルタでカットされる恐れがあるため使用しない。
PHONE_FREQS = [523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77]

# 同期用プリアンブル: どちらのモードでも「トニックの1オクターブ上」を使う。
# データの7音とは重ならないので、受信側はこの音を目印にブロック境界を特定できる。
# 実機ではスピーカーがサスペンド状態から復帰する際に先頭が欠落することがあるため、
# 必要な2ブロック分より長めに(余裕を持って)鳴らす。
SYNC_DURATION_BLOCKS = 4  # 1シンボル分の何個ぶんの長さの同期音にするか

# 同期音の直後にスピーカー/部屋の残響が乗ったまま最初のデータ音が始まると、
# 受信側で2つの周波数が混ざって最初の1〜2シンボルが誤検出されやすい。
# 短い無音区間を挟んで残響を減衰させてから最初のデータ音を鳴らす。
SYNC_GAP = 0.05  # seconds

# find_sync_startの境界検出には数十ms単位の誤差が残り得る。特にファイルの末尾
# ぴったりでデータが終わっていると、その誤差分だけデコード側が範囲外を切り捨てて
# 最後のシンボルが丸ごと失われる。末尾に余裕を持たせておくことで吸収する。
TAIL_PAD = 0.1  # seconds


def sync_freq_for(freqs: list[float]) -> float:
    return freqs[0] * 2


def text_to_symbols(text: str) -> list[int]:
    symbols = []
    for byte in text.encode("utf-8"):
        n = byte
        digits = []
        for _ in range(3):
            digits.append(n % 7)
            n //= 7
        symbols.extend(reversed(digits))  # 上位桁から
    return symbols


def symbols_to_text(symbols: list[int]) -> str:
    raw = bytearray()
    complete_len = len(symbols) - len(symbols) % 3  # 末尾の不完全な端数(ノイズ由来)は捨てる
    for i in range(0, complete_len, 3):
        n = 0
        for d in symbols[i:i + 3]:
            n = n * 7 + d
        raw.append(n & 0xFF)
    return raw.decode("utf-8", errors="replace")


def synth_tone(freq: float, duration: float, fade_ms: float = FADE_MS) -> np.ndarray:
    n_samples = int(SAMPLE_RATE * duration)
    t = np.arange(n_samples) / SAMPLE_RATE
    wave_data = 0.6 * np.sin(2 * np.pi * freq * t)

    fade_samples = int(SAMPLE_RATE * fade_ms / 1000)
    envelope = np.ones(n_samples)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return wave_data * envelope


def synth_symbol(digit: int, symbol_duration: float = SYMBOL_DURATION,
                  fade_ms: float = FADE_MS, freqs: list[float] = FREQS) -> np.ndarray:
    return synth_tone(freqs[digit], symbol_duration, fade_ms)


def encode_to_wav(text: str, path: str, symbol_duration: float = SYMBOL_DURATION,
                   fade_ms: float = FADE_MS, freqs: list[float] = FREQS) -> list[int]:
    symbols = text_to_symbols(text)
    sync = synth_tone(sync_freq_for(freqs), symbol_duration * SYNC_DURATION_BLOCKS, fade_ms)
    gap = np.zeros(int(SAMPLE_RATE * SYNC_GAP))
    tail = np.zeros(int(SAMPLE_RATE * TAIL_PAD))
    data = np.concatenate([synth_symbol(s, symbol_duration, fade_ms, freqs) for s in symbols])
    audio = np.concatenate([sync, gap, data, tail])
    pcm = np.int16(audio * 32767)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return symbols


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    phone_mode = "--phone" in args
    if phone_mode:
        args.remove("--phone")

    message = "HELLO"
    duration = float(args[0]) if len(args) > 0 else SYMBOL_DURATION
    out_path = args[1] if len(args) > 1 else ("sample_phone.wav" if phone_mode else "sample.wav")
    fade_ms = min(FADE_MS, duration * 1000 * 0.1)  # 1音の1割を上限にフェードを短縮
    freqs = PHONE_FREQS if phone_mode else FREQS

    symbols = encode_to_wav(message, out_path, duration, fade_ms, freqs)
    notes = " ".join(NOTE_NAMES[s] for s in symbols)
    print(f"メッセージ: {message}")
    print(f"モード: {'電話回線用(C5-B5)' if phone_mode else '通常(C5-B5)'}")
    print(f"シンボル列（7進数）: {symbols}")
    print(f"音階列: {notes}")
    print(f"1音の長さ: {duration * 1000:.0f}ms  フェード: {fade_ms:.1f}ms")
    print(f"音の数: {len(symbols)}  再生時間: {len(symbols) * duration:.2f}秒")
    print(f"出力: {out_path}")
