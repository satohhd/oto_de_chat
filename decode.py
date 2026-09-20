"""音声(wav/生データ)をFFTで解析し、シンボル区間ごとの音階から元のメッセージを復元する。"""
import wave
import numpy as np

from encode import (
    SAMPLE_RATE, SYMBOL_DURATION, FREQS, PHONE_FREQS, NOTE_NAMES,
    SYNC_DURATION_BLOCKS, SYNC_GAP, sync_freq_for, symbols_to_text,
)

# 同期音とみなす周波数の許容誤差。FREQS/PHONE_FREQSはオクターブが違っても音階の
# 相対的な間隔(比率)は同じなので、絶対Hzではなく同期音周波数に対する比率で持つ。
# こうすることで、音階テーブルの音域を(オクターブを変えるなどして)変更しても
# 同じ相対的なドリフト耐性を保てる。
SYNC_TOLERANCE_RATIO = 0.029  # 同期音とみなす周波数の許容誤差(クリーンな信号での通常ロック用)
# ドリフトが大きい信号は上のSYNC_TOLERANCE_RATIOでロックできないことがあるため、
# ロック失敗時はこちらの緩い許容誤差で再試行する。データ音の最高周波数(シ)との
# 間隔(比率で約5.6%)は超えないようにしてある。
SYNC_TOLERANCE_RATIO_WIDE = 0.048
MAX_SYNC_DRIFT = 0.08  # これを超えるズレは測定ミス(ノイズ等)とみなし補正なしにフォールバック


def peak_freq(block: np.ndarray, interpolate: bool = False) -> float:
    spectrum = np.abs(np.fft.rfft(block))
    freqs_axis = np.fft.rfftfreq(len(block), d=1 / SAMPLE_RATE)
    k = np.argmax(spectrum)
    if not interpolate or k == 0 or k == len(spectrum) - 1:
        return freqs_axis[k]
    # 放物線補間: FFTのビン幅(SAMPLE_RATE/N ≈ 4Hz)より細かい精度でピーク周波数を
    # 推定する。同期音からのドリフト比推定にのみ使う(データシンボルの判定は
    # 従来通りビン単位のままでよく、既存のnearest-neighbor判定を変えない)。
    a, b, c = spectrum[k - 1], spectrum[k], spectrum[k + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return freqs_axis[k]
    offset = 0.5 * (a - c) / denom
    bin_width = freqs_axis[1] - freqs_axis[0]
    return freqs_axis[k] + offset * bin_width


def goertzel_magnitude(block: np.ndarray, freq: float) -> float:
    """blockの中に周波数freqの成分がどれだけ含まれるかを表す量(相関の大きさ)。
    単一周波数への直接相関(Goertzelアルゴリズムと数学的に同じ結果を与える)。
    FFTのビン格子に量子化されないため、durationやドリフト補正でfreqがどんな値に
    なっても劣化しない。
    """
    n = np.arange(len(block))
    basis = np.exp(-2j * np.pi * freq * n / SAMPLE_RATE)
    return abs(np.sum(block * basis))


def tone_magnitudes(block: np.ndarray, freqs: list[float], ratio: float = 1.0) -> np.ndarray:
    """freqsの各候補周波数(ratio補正込み)について、blockとの相関の大きさをまとめて返す。"""
    n = np.arange(len(block))
    targets = np.asarray(freqs) * ratio
    basis = np.exp(-2j * np.pi * np.outer(targets, n) / SAMPLE_RATE)  # shape (K, N)
    return np.abs(basis @ block)


def decode_block(block: np.ndarray, freqs: list[float], ratio: float = 1.0) -> int:
    return int(np.argmax(tone_magnitudes(block, freqs, ratio)))


SYNC_LOCK_BLOCKS = 2  # 同期とみなすために連続一致を要求するブロック数(実際の送信長より短くてよい)


def _is_sync_block(block: np.ndarray, sync_freq: float, tolerance: float) -> bool:
    return abs(peak_freq(block) - sync_freq) <= tolerance


def _find_lock_offset(audio: np.ndarray, block_samples: int, sync_freq: float,
                       max_offset: int, step: int, tolerance: float) -> int | None:
    for offset in range(0, max_offset + 1, step):
        blocks = [audio[offset + i * block_samples: offset + (i + 1) * block_samples]
                  for i in range(SYNC_LOCK_BLOCKS)]
        if all(_is_sync_block(b, sync_freq, tolerance) for b in blocks):
            return offset
    return None


def find_sync_start(audio: np.ndarray, symbol_duration: float, freqs: list[float],
                     search_samples: int | None = None) -> tuple[int, float] | None:
    """audioの先頭付近を走査し、同期音(トニックの1オクターブ上)を検出したら、
    (データ開始位置, ドリフト比)を返す。見つからなければNone。

    実機ではスピーカーの起動遅延等で同期音の先頭が欠落することがあるため、
    「決まった長さぶん一致するか」ではなく「一致し続ける限り読み進め、
    途切れた瞬間をデータ開始点とみなす」方式にして、欠落に強くしている。

    ドリフト比は、送受信機のオーディオクロックが公称のSAMPLE_RATEからズレている場合に
    全周波数が比例してシフトする現象を補正するためのもの。同期音の理論周波数は既知
    (sync_freq)なので、ロックしたブロックの実測ピーク周波数との比を測ればそのまま
    データ音の判定境界の補正係数として使い回せる。
    """
    block_samples = int(SAMPLE_RATE * symbol_duration)
    lock_samples = block_samples * SYNC_LOCK_BLOCKS
    sync_freq = sync_freq_for(freqs)

    if search_samples is None:
        search_samples = block_samples
    step = max(1, int(SAMPLE_RATE * 0.01))  # 10ms刻みでロック位置候補を探索
    max_offset = min(search_samples, len(audio) - lock_samples)
    if max_offset < 0:
        return None

    strict_tolerance = sync_freq * SYNC_TOLERANCE_RATIO
    wide_tolerance = sync_freq * SYNC_TOLERANCE_RATIO_WIDE

    lock_offset = _find_lock_offset(audio, block_samples, sync_freq, max_offset, step, strict_tolerance)
    tolerance = strict_tolerance
    if lock_offset is None:
        # 通常の許容誤差でロックできない場合、送受信機のクロックドリフトで同期音自体が
        # ズレている可能性があるので、緩い許容誤差で再試行する(ロックできればこの後の
        # ドリフト比測定で実際のズレ量を求め、データ音の判定に補正をかけられる)。
        lock_offset = _find_lock_offset(audio, block_samples, sync_freq, max_offset, step, wide_tolerance)
        tolerance = wide_tolerance

    if lock_offset is None:
        return None

    # ロックした同期音ブロックの実測ピーク周波数(サブビン補間つき)からドリフト比を推定する
    sync_peaks = [
        peak_freq(audio[lock_offset + i * block_samples: lock_offset + (i + 1) * block_samples], interpolate=True)
        for i in range(SYNC_LOCK_BLOCKS)
    ]

    # ロックできたら、同期音が途切れるまでブロック単位で読み進める(粗い境界)
    pos = lock_offset + lock_samples
    while pos + block_samples <= len(audio) and _is_sync_block(audio[pos:pos + block_samples], sync_freq, tolerance):
        sync_peaks.append(peak_freq(audio[pos:pos + block_samples], interpolate=True))
        pos += block_samples

    ratio = float(np.mean(sync_peaks)) / sync_freq
    if not (1 - MAX_SYNC_DRIFT <= ratio <= 1 + MAX_SYNC_DRIFT):
        ratio = 1.0  # 信頼できない測定は補正なし(従来どおり)にフォールバック

    # 上のブロック単位の判定では、同期音とデータの継ぎ目がブロック(=1シンボル分、
    # 実機では最大250ms相当)ずれることがある。ずれたままだと以降の全シンボルが
    # 隣接する2音の周波数が混ざった状態でデコードされてしまうため、継ぎ目付近を
    # 10ms刻みの短い窓で細かく走査し、実際に同期音が途切れる位置を追い直す。
    fine_window = min(block_samples, int(SAMPLE_RATE * 0.1))
    fine_step = max(1, int(SAMPLE_RATE * 0.01))
    refined = pos
    p = max(lock_offset + lock_samples, pos - block_samples)
    while p + fine_window <= len(audio) and p <= pos + block_samples:
        if _is_sync_block(audio[p:p + fine_window], sync_freq, tolerance):
            # fine_window全体のピーク周波数で判定しているため、窓の半分が同期音・
            # 半分がデータ音になった時点まで「まだ同期音」と判定され続ける(多数決)。
            # そのため境界はp+fine_windowではなく、その半分の位置とみなすのが正しい。
            refined = p + fine_window // 2
        p += fine_step

    # 同期音とデータの間には残響よけの無音(SYNC_GAP)を挟んであるので、
    # 検出した「同期音が途切れる位置」からさらにその分進めた所がデータの先頭。
    return refined + int(SAMPLE_RATE * SYNC_GAP), ratio


def decode_audio(audio: np.ndarray, symbol_duration: float = SYMBOL_DURATION,
                  freqs: list[float] = FREQS, ratio: float = 1.0) -> list[int]:
    block_size = int(SAMPLE_RATE * symbol_duration)
    n_blocks = len(audio) // block_size
    return [decode_block(audio[i * block_size:(i + 1) * block_size], freqs, ratio) for i in range(n_blocks)]


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "r") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767


def decode_wav(path: str, symbol_duration: float = SYMBOL_DURATION,
                freqs: list[float] = FREQS) -> tuple[list[int], str, float]:
    audio = read_wav(path)
    result = find_sync_start(audio, symbol_duration, freqs)
    # 同期音が見つからない場合は先頭からそのままデコード(後方互換)、補正もなし
    start, ratio = result if result is not None else (0, 1.0)
    symbols = decode_audio(audio[start:], symbol_duration, freqs, ratio)
    text = symbols_to_text(symbols)
    return symbols, text, ratio


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    phone_mode = "--phone" in args
    if phone_mode:
        args.remove("--phone")

    path = args[0] if len(args) > 0 else "sample.wav"
    duration = float(args[1]) if len(args) > 1 else SYMBOL_DURATION
    freqs = PHONE_FREQS if phone_mode else FREQS

    symbols, text, ratio = decode_wav(path, duration, freqs)
    notes = " ".join(NOTE_NAMES[s] for s in symbols)
    print(f"モード: {'電話回線用(C5-B5)' if phone_mode else '通常(C5-B5)'}")
    print(f"検出したドリフト比: {ratio:.4f}")
    print(f"検出した音階列: {notes}")
    print(f"シンボル列（7進数）: {symbols}")
    print(f"復元したメッセージ: {text}")
