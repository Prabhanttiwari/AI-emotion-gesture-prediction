import cv2
import numpy as np
from fer import FER
from PIL import Image
import os
import time

ASSETS_DIR = "assets"  # folder for emotion PNGs

EMO_TO_ASSET = {
    "happy": "happy.png",
    "sad": "sad.png",
    "angry": "angry.png",
    "surprise": "surprise.png",
    "neutral": "neutral.png",
    "fear": "fear.png",
    "disgust": "disgust.png"
}

def load_assets(dirpath):
    assets = {}
    for emo, fname in EMO_TO_ASSET.items():
        path = os.path.join(dirpath, fname)
        if os.path.exists(path):
            assets[emo] = Image.open(path).convert("RGBA")
        else:
            assets[emo] = None
    return assets

assets = load_assets(ASSETS_DIR)

def overlay_pil_on_cv2(frame_bgr, pil_img, x, y, w, h, alpha_scale=1.0):
    if pil_img is None:
        return frame_bgr
    pil_resized = pil_img.resize((max(1, int(w)), max(1, int(h))), resample=Image.LANCZOS)
    overlay_np = np.array(pil_resized)
    if overlay_np.shape[2] == 3:
        b, g, r = cv2.split(overlay_np)
        a = np.ones(b.shape, dtype=np.uint8) * 255
    else:
        b, g, r, a = cv2.split(overlay_np)
    overlay_bgr = cv2.merge([b, g, r])
    mask = (a.astype(float) / 255.0) * alpha_scale

    fh, fw = frame_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)

    if x1 >= x2 or y1 >= y2:
        return frame_bgr

    ox1, oy1 = max(0, -x), max(0, -y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    roi = frame_bgr[y1:y2, x1:x2].astype(float)
    ov = overlay_bgr[oy1:oy2, ox1:ox2].astype(float)
    m = mask[oy1:oy2, ox1:ox2][:, :, None]

    blended = (ov * m + roi * (1 - m)).astype(np.uint8)
    frame_bgr[y1:y2, x1:x2] = blended
    return frame_bgr


# Initialize FER detector (offline)
emotion_detector = FER(mtcnn=False)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

ANALYZE_EVERY_N_FRAMES = 6
frame_count = 0
last_result = None
last_time = 0

print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    frame_count += 1

    do_analyze_now = (frame_count % ANALYZE_EVERY_N_FRAMES == 0)
    if do_analyze_now:
        try:
            results = emotion_detector.detect_emotions(frame)
            if results:
                result = results[0]
                last_result = result
                last_time = time.time()
        except Exception as e:
            pass

    if last_result is not None:
        box = last_result["box"]
        emotions = last_result["emotions"]
        x, y, w_box, h_box = box
        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]

        cv2.rectangle(display, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        # smaller persona overlay
        scale = 0.6
        pad = int(0.05 * max(w_box, h_box))
        ow = int(w_box * scale)
        oh = int(h_box * scale)
        ox = x + (w_box - ow) // 2
        oy = y - oh - 10

        asset_key = dominant if dominant in assets else "neutral"
        persona_img = assets.get(asset_key, None)
        display = overlay_pil_on_cv2(display, persona_img, ox, oy, ow, oh, alpha_scale=0.95)

        text = f"{dominant} ({int(confidence * 100)}%)"
        cv2.putText(display, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    now = time.time()
    if last_time:
        age = now - last_time
        cv2.putText(display, f"Last analysis: {age:.1f}s ago", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("AI Face Emotion Persona Overlay (Offline FER)", display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
