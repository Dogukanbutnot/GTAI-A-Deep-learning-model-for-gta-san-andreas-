"""
collect_data.py
Ekran görüntüsü + klavye girdisini eş zamanlı kaydeder.

Kullanım:
    python collect_data.py --session surus_okulu_01 --fps 30

Kayıt formatı:
    data/sessions/<session_name>/
        frames/
            000000.npy   ← (224, 224, 3) float32
            000001.npy
            ...
        labels.npy       ← (N,) int64  action label'ları
        meta.json        ← oturum bilgileri
"""

import argparse
import json
import time
import threading
from pathlib import Path

import numpy as np
import cv2

from screen_capture import create_capturer, grab_frame, preprocess_frame
from input_logger   import start_listener, stop_listener, get_action, action_to_label


# ─────────────────────────────────────────
# Ayarlar
# ─────────────────────────────────────────
BASE_DIR    = Path("data/sessions")
DEFAULT_FPS = 30


def collect(session_name: str, fps: int):
    """Ana kayıt döngüsü."""
    out_dir    = BASE_DIR / session_name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_interval = 1.0 / fps
    sct, monitor   = create_capturer()
    start_listener()

    labels     = []
    frame_idx  = 0
    t_session  = time.time()

    print(f"\n[Collector] Oturum: {session_name}")
    print(f"[Collector] FPS hedef: {fps}")
    print(f"[Collector] Çıkmak için 'q' (önizleme) veya Ctrl+C\n")

    # Geri sayım — oyuna geçiş için
    for i in range(3, 0, -1):
        print(f"  Başlıyor... {i}", end="\r", flush=True)
        time.sleep(1)
    print("  KAYIT BAŞLADI!         ")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Ekranı yakala
            raw_frame  = grab_frame(sct, monitor)
            proc_frame = preprocess_frame(raw_frame)

            # 2. Aynı anda tuş durumunu al
            action = get_action()
            label  = action_to_label(action)

            # 3. Kaydet
            np.save(frames_dir / f"{frame_idx:06d}.npy", proc_frame)
            labels.append(label)

            # 4. Önizleme
            preview = cv2.resize(raw_frame, (640, 360))
            w, a, s, d, space = action
            info = (f"Frame:{frame_idx:5d}  Label:{label}  "
                    f"[W={w} A={a} S={s} D={d} SPC={space}]")
            cv2.putText(preview, info, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
            cv2.imshow("GTA SA — Data Collector", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[Collector] 'q' ile durduruldu.")
                break

            frame_idx += 1

            # FPS limitle
            elapsed = time.perf_counter() - t0
            wait    = frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        print("\n[Collector] Ctrl+C ile durduruldu.")

    finally:
        cv2.destroyAllWindows()
        stop_listener()

        # labels.npy kaydet
        labels_arr = np.array(labels, dtype=np.int64)
        np.save(out_dir / "labels.npy", labels_arr)

        # meta.json kaydet
        duration = time.time() - t_session
        meta = {
            "session":    session_name,
            "fps":        fps,
            "frames":     frame_idx,
            "duration_s": round(duration, 2),
            "label_dist": {str(i): int(np.sum(labels_arr == i)) for i in range(10)},
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Özet
        print(f"\n{'─'*50}")
        print(f"  Oturum tamamlandı: {session_name}")
        print(f"  Toplam frame : {frame_idx}")
        print(f"  Süre         : {duration:.1f} sn")
        print(f"  Kayıt yeri   : {out_dir}")
        print(f"  Action dağılımı:")
        action_names = ["Boşta","İleri","Sol","Sağ","Geri","İleri+Sol",
                        "İleri+Sağ","Geri+Sol","Geri+Sağ","ElFreni"]
        for i, name in enumerate(action_names):
            count = meta["label_dist"][str(i)]
            bar   = "█" * (count // max(frame_idx // 40, 1))
            print(f"  {i}: {name:<12} {bar} ({count})")
        print(f"{'─'*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTA SA Veri Toplayıcı")
    parser.add_argument("--session", type=str, required=True,
                        help="Oturum adı (örn: surus_okulu_01)")
    parser.add_argument("--fps",     type=int, default=DEFAULT_FPS,
                        help=f"Hedef FPS (varsayılan: {DEFAULT_FPS})")
    args = parser.parse_args()

    collect(session_name=args.session, fps=args.fps)
