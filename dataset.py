"""
input_logger.py
Klavye girişlerini kaydeder ve hangi tuşların basılı olduğunu döndürür.

GTA SA sürüş kontrolleri:
  W → ileri (gaz)
  S → geri / fren
  A → sola dön
  D → sağa dön
  Boşluk → el freni

Action encoding (one-hot değil, multi-hot):
  [W, A, S, D, SPACE] → her biri 0 veya 1
  Örnek: [1, 0, 0, 1, 0] → W + D (gaz + sağ)
"""

import time
import threading
import numpy as np
from pynput import keyboard


# Takip edilecek tuşlar ve indeksleri
KEYS = {
    "w":     0,
    "a":     1,
    "s":     2,
    "d":     3,
    "space": 4,
}

# Paylaşılan durum (thread-safe için lock ile korunuyor)
_pressed = {k: False for k in KEYS}
_lock    = threading.Lock()


def _on_press(key):
    """pynput callback — tuş basıldığında."""
    try:
        char = key.char.lower() if hasattr(key, "char") else None
    except AttributeError:
        char = None

    # Boşluk tuşu özel işlem
    if key == keyboard.Key.space:
        with _lock:
            _pressed["space"] = True
        return

    if char in KEYS:
        with _lock:
            _pressed[char] = True


def _on_release(key):
    """pynput callback — tuş bırakıldığında."""
    try:
        char = key.char.lower() if hasattr(key, "char") else None
    except AttributeError:
        char = None

    if key == keyboard.Key.space:
        with _lock:
            _pressed["space"] = False
        return

    if char in KEYS:
        with _lock:
            _pressed[char] = False


# Global listener (bir kez başlatılır)
_listener = None


def start_listener():
    """Arka planda klavye dinleyicisini başlatır."""
    global _listener
    if _listener is None:
        _listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        _listener.daemon = True
        _listener.start()
        print("[InputLogger] Klavye dinleyicisi başlatıldı.")


def stop_listener():
    """Klavye dinleyicisini durdurur."""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None
        print("[InputLogger] Klavye dinleyicisi durduruldu.")


def get_action() -> np.ndarray:
    """
    Anlık tuş durumunu döndürür.
    Çıktı: shape (5,) int8 array → [W, A, S, D, SPACE]
    """
    with _lock:
        action = np.array(
            [int(_pressed[k]) for k in ["w", "a", "s", "d", "space"]],
            dtype=np.int8,
        )
    return action


def action_to_label(action: np.ndarray) -> int:
    """
    5-bit multi-hot vektörü tek bir sınıf etiketine çevirir.
    CNN için basit discrete action space:

      0 → Hiçbir şey (boşta)
      1 → İleri (W)
      2 → Sola (A)
      3 → Sağa (D)
      4 → Fren/Geri (S)
      5 → İleri + Sola (W+A)
      6 → İleri + Sağa (W+D)
      7 → Fren + Sola (S+A)
      8 → Fren + Sağa (S+D)
      9 → El freni (SPACE)
    """
    w, a, s, d, space = action

    if space:           return 9
    if w and a:         return 5
    if w and d:         return 6
    if s and a:         return 7
    if s and d:         return 8
    if w:               return 1
    if a:               return 2
    if d:               return 3
    if s:               return 4
    return 0  # boşta


def label_to_keys(label: int) -> dict:
    """
    Sınıf etiketini tekrar tuş sözlüğüne çevirir.
    AI'ın kararını oyuna uygulamak için kullanılır.
    Dönüş: {"w": bool, "a": bool, "s": bool, "d": bool, "space": bool}
    """
    mapping = {
        0: (0, 0, 0, 0, 0),
        1: (1, 0, 0, 0, 0),
        2: (0, 1, 0, 0, 0),
        3: (0, 0, 0, 1, 0),
        4: (0, 0, 1, 0, 0),
        5: (1, 1, 0, 0, 0),
        6: (1, 0, 0, 1, 0),
        7: (0, 1, 1, 0, 0),
        8: (0, 0, 1, 1, 0),
        9: (0, 0, 0, 0, 1),
    }
    w, a, s, d, space = mapping.get(label, (0, 0, 0, 0, 0))
    return {"w": bool(w), "a": bool(a), "s": bool(s), "d": bool(d), "space": bool(space)}


# --- Test ---
if __name__ == "__main__":
    start_listener()
    print("Tuşlara bas, çıkmak için Ctrl+C")
    try:
        while True:
            action = get_action()
            label  = action_to_label(action)
            print(f"\r[W={action[0]} A={action[1]} S={action[2]} D={action[3]} SPC={action[4]}]"
                  f"  Label={label:2d}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        stop_listener()
