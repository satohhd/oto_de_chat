"""マイク入力を常時ストリーミングで監視し、同期音(プリアンブル)を検出して正確な
ブロック境界を特定してから逐次デコードする(録音ファイルを介さないリアルタイム版)。

使い方:
    python3 live_decode.py [1音の長さ(秒)] [--phone]

Ctrl+Cで終了。
"""
import subprocess
import sys

import numpy as np

from encode import SAMPLE_RATE, SYMBOL_DURATION, FREQS, PHONE_FREQS, SYNC_DURATION_BLOCKS, symbols_to_text
from decode import decode_block, find_sync_start

HOP_MS = 10  # 無音検出の時間分解能
END_SILENCE_RATIO = 1.5  # 1音の長さの何倍、無音が続いたらメッセージ終了とみなすか
SYNC_SEARCH_BLOCKS = 1.5  # onset検出後、同期音を探す範囲(ブロック数)

# ブロック(250ms)単位で「これは本物の音か、無音/残響か」を判定する際のしきい値の割合。
# onset検出用のthreshold(ノイズフロアの4倍)をそのまま使うと、スピーカー/マイクの
# 距離や機種によって受信音量とノイズフロアの比(SNR)が小さいPCでは、正規のデータ音
# まで「残響」とみなして捨ててしまう(受信中の無音判定はEND_SILENCE_RATIOで別途行う
# ので、ここは緩めでよい)。
BLOCK_GATE_RATIO = 0.5


def start_stream():
    return subprocess.Popen(
        ["arecord", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
    )


def read_exact(stream, n_bytes: int) -> bytes:
    data = b""
    while len(data) < n_bytes:
        chunk = stream.stdout.read(n_bytes - len(data))
        if not chunk:
            raise EOFError("audio stream ended")
        data += chunk
    return data


def to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32767


def main() -> None:
    args = sys.argv[1:]
    phone_mode = "--phone" in args
    if phone_mode:
        args.remove("--phone")
    duration = float(args[0]) if args else SYMBOL_DURATION
    freqs = PHONE_FREQS if phone_mode else FREQS

    hop_samples = int(SAMPLE_RATE * HOP_MS / 1000)
    block_samples = int(SAMPLE_RATE * duration)
    search_samples = int(block_samples * SYNC_SEARCH_BLOCKS)

    mode_label = "電話回線用(C5-B5)" if phone_mode else "通常(C5-B5)"
    print(f"[リアルタイム受信] モード:{mode_label} 1音:{duration * 1000:.0f}ms (Ctrl+Cで終了)")

    stream = start_stream()
    try:
        # 録音デバイスによっては、ストリームを開いた直後の数百msに
        # (サスペンドからの復帰時のポップノイズ等で)フルスケールに近い
        # 過渡的なノイズが乗ることがある。これを較正に含めるとノイズレベルを
        # 大幅に過大評価してしまうため、最初の1秒は捨ててから較正する。
        read_exact(stream, hop_samples * 2 * 100)

        # 続く0.3秒で環境ノイズレベルを推定
        calib = to_float(read_exact(stream, hop_samples * 2 * 30))
        noise_rms = np.sqrt(np.mean(calib.reshape(-1, hop_samples) ** 2, axis=1)).mean()
        threshold = max(noise_rms * 4, 1e-4)
        print(f"(ノイズレベル推定: {noise_rms:.5f}  検出しきい値: {threshold:.5f})")
        print("待機中...")

        # IDLE: 無音を監視。onsetを検知したらBUFFERING(同期音を探索中)へ。
        # BUFFERING: 一定量たまったら同期音を探索し、見つかればACTIVEへ。見つからなければIDLEへ。
        # ACTIVE: 1ブロックずつデコードしながら、無音が続いたらメッセージ確定してIDLEへ。
        state = "IDLE"
        buffer = np.array([], dtype=np.float64)
        symbols: list[int] = []
        silence_run = 0

        while True:
            hop = to_float(read_exact(stream, hop_samples * 2))
            rms = np.sqrt(np.mean(hop ** 2))

            if state == "IDLE":
                if rms >= threshold:
                    state = "BUFFERING"
                    buffer = hop.copy()
                    print("[信号検出、同期音を探索中...]")
                continue

            if state == "BUFFERING":
                buffer = np.concatenate([buffer, hop])
                # 探索範囲の余裕 + 実際に送信される同期音の全長(SYNC_DURATION_BLOCKS)
                # + 境界を細かく追い直すための余裕1ブロック分をあらかじめ確保しておく。
                # これが足りないと、同期音が途切れる位置を見つける前にバッファの端に
                # 達してしまい、データ開始位置の検出がズレる。
                needed = search_samples + block_samples * (SYNC_DURATION_BLOCKS + 2)
                if len(buffer) < needed:
                    continue

                result = find_sync_start(buffer, duration, freqs, search_samples=search_samples)
                if result is None:
                    print("[同期音が見つからず、ノイズと判断して破棄]")
                    state = "IDLE"
                    buffer = np.array([], dtype=np.float64)
                    continue

                start, ratio = result
                print(f"[同期音を検出、データ受信開始](ドリフト比: {ratio:.4f})")
                buffer = buffer[start:]
                symbols = []
                silence_run = 0
                state = "ACTIVE"
                # このままACTIVE側の処理に落として、既にたまっているbufferを即座に消費する

            if state == "ACTIVE":
                buffer = np.concatenate([buffer, hop])
                # 受信中の無音判定も、ブロック採否と同じ緩めのしきい値で行う。
                # onset検出用のthresholdをそのまま使うと、スピーカー/マイクの
                # 距離や機種によってSNRが小さいPCでは受信中の正規のデータ音まで
                # 「無音」とみなしてしまい、メッセージが途中で打ち切られる。
                active_threshold = threshold * BLOCK_GATE_RATIO
                silence_run = silence_run + 1 if rms < active_threshold else 0

                while len(buffer) >= block_samples:
                    block = buffer[:block_samples]
                    buffer = buffer[block_samples:]
                    # ブロック自体の音量がしきい値未満なら、音の余韻(残響)や
                    # 無音とみなし、ノイズシンボルとして拾わない
                    if np.sqrt(np.mean(block ** 2)) >= active_threshold:
                        symbols.append(decode_block(block, freqs, ratio))

                if silence_run * HOP_MS >= duration * 1000 * END_SILENCE_RATIO:
                    if symbols:
                        print(f">> {symbols_to_text(symbols)!r}")
                    state = "IDLE"
                    buffer = np.array([], dtype=np.float64)
                    symbols = []
                    silence_run = 0
                    print("待機中...")

    except KeyboardInterrupt:
        print("\n終了します")
    finally:
        stream.terminate()


if __name__ == "__main__":
    main()
