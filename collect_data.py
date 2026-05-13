"""
screen_capture.py
Oyun ekranını gerçek zamanlı olarak yakalar.
Kullanım: python screen_capture.py
"""

import mss
import numpy as np
import cv2
import time
from pathlib import Path


# --- AYARLAR ---
# GTA SA penceresinin ekrandaki konumu (piksel cinsinden)
# Oyunu açıp tam ekran yapınca genellikle (0,0) olur.
# Değilse, oyun penceresini ölçüp buraya girin.
CAPTURE_REGION = {
    "top": 0,
    "left": 0,
    "width": 1280,
    "height": 720,
}

TARGET_SIZE = (224, 224)   # CNN'e girecek boyut
FPS_LIMIT   = 30           # Saniyede kaç frame yakalanacak
PREVIEW     = True         # True ise küçük bir önizleme penceresi açar


def create_capturer(region: dict = CAPTURE_REGION):
    """MSS ekran yakalayıcıyı başlatır."""
    sct = mss.mss()
    monitor = region
    return sct, monitor


def grab_frame(sct, monitor) -> np.ndarray:
    """
    Tek bir frame yakalar.
    Çıktı: (H, W, 3) şeklinde BGR numpy array (OpenCV formatı).
    """
    raw = sct.grab(monitor)
    # mss BGRA döndürür → BGR'ye çevir
    frame = np.array(raw)[:, :, :3]
    return frame


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Frame'i CNN'e hazırlar:
      1. Hedef boyuta küçült
      2. [0, 1] aralığına normalize et
    Çıktı: (224, 224, 3) float32
    """
    resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def capture_loop(save_raw: bool = False, output_dir: str = "data/raw"):
    """
    Sürekli ekran yakalar.
    save_raw=True ise ham frame'leri diske kaydeder (veri toplama sırasında kullanılmaz,
    sadece debug için).
    """
    if save_raw:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    sct, monitor = create_capturer()
    frame_interval = 1.0 / FPS_LIMIT
    frame_count = 0

    print(f"[ScreenCapture] Başlatıldı — {FPS_LIMIT} FPS hedef, bölge: {monitor}")
    print("[ScreenCapture] Çıkmak için 'q' tuşuna basın (önizleme açıksa).")

    try:
        while True:
            t_start = time.perf_counter()

            frame = grab_frame(sct, monitor)
            processed = preprocess_frame(frame)

            if PREVIEW:
                preview = cv2.resize(frame, (640, 360))
                cv2.putText(preview, f"Frame: {frame_count}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("GTA SA - Capture Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if save_raw:
                path = f"{output_dir}/frame_{frame_count:06d}.png"
                cv2.imwrite(path, frame)

            frame_count += 1

            # FPS limitle
            elapsed = time.perf_counter() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[ScreenCapture] Durduruldu.")
    finally:
        cv2.destroyAllWindows()
        print(f"[ScreenCapture] Toplam yakalanan frame: {frame_count}")


if __name__ == "__main__":
    capture_loop(save_raw=False)
