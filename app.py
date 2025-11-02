import streamlit as st
import cv2
import numpy as np
from fer import FER
from PIL import Image
import os

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="AI Emotion Detector", page_icon="🤖", layout="centered")

# ---------- LOAD EXTERNAL CSS ----------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# ---------- JS DRAG LOGIC ----------
st.markdown("""
<script>
var dragItem = null;
document.addEventListener("mousedown", function(e) {
  if (e.target.closest("#draggable-div")) {
    dragItem = e.target.closest("#draggable-div");
    offsetX = e.clientX - dragItem.getBoundingClientRect().left;
    offsetY = e.clientY - dragItem.getBoundingClientRect().top;
  }
});
document.addEventListener("mouseup", function() {
  dragItem = null;
});
document.addEventListener("mousemove", function(e) {
  if (dragItem) {
    dragItem.style.left = (e.clientX - offsetX) + "px";
    dragItem.style.top = (e.clientY - offsetY) + "px";
  }
});
</script>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>AI EMOTION DETECTOR</h1>", unsafe_allow_html=True)
st.markdown("<h2>Built by Prabhant Tiwari 🤖</h2>", unsafe_allow_html=True)

# ---------- LOAD EMOTION ICONS ----------
ASSETS_DIR = "assets"
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

# ---------- HELPER ----------
def overlay_pil_on_cv2(frame_bgr, pil_img, x, y, w, h):
    if pil_img is None:
        return frame_bgr
    pil_resized = pil_img.resize((w, h), resample=Image.LANCZOS)
    overlay_np = np.array(pil_resized)
    if overlay_np.shape[2] == 3:
        b, g, r = cv2.split(overlay_np)
        a = np.ones(b.shape, dtype=np.uint8) * 255
    else:
        b, g, r, a = cv2.split(overlay_np)
    overlay_bgr = cv2.merge([b, g, r])
    mask = a.astype(float) / 255.0

    h_, w_ = frame_bgr.shape[:2]
    if y + h > h_ or x + w > w_:
        return frame_bgr
    roi = frame_bgr[y:y + h, x:x + w]
    blended = (overlay_bgr * mask[..., None] + roi * (1 - mask[..., None])).astype(np.uint8)
    frame_bgr[y:y + h, x:x + w] = blended
    return frame_bgr

# ---------- APP CARD ----------
st.markdown('<div class="card"><p>Detects emotions in real-time using your webcam — fully offline </p></div>', unsafe_allow_html=True)

# ---------- BUTTON ----------
start = st.button("🎥 Start Emotion Detection")

if start:
    st.markdown('<div id="draggable-div">', unsafe_allow_html=True)
    FRAME_WINDOW = st.image([], channels="BGR", width=480)
    st.markdown('</div>', unsafe_allow_html=True)

    emotion_detector = FER(mtcnn=False)
    cap = cv2.VideoCapture(0)
    st.write("**Camera started! Drag the webcam window anywhere 👆**")

    last_result = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Cannot access webcam.")
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        frame_count += 1

        if frame_count % 6 == 0:
            results = emotion_detector.detect_emotions(frame)
            if results:
                last_result = results[0]

        if last_result:
            box = last_result["box"]
            emotions = last_result["emotions"]
            x, y, w_box, h_box = box
            dominant = max(emotions, key=emotions.get)
            conf = emotions[dominant]

            cv2.rectangle(display, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            ow, oh = int(w_box * 0.4), int(h_box * 0.4)
            ox, oy = x + (w_box - ow) // 2, y - oh - 10

            emoji = assets.get(dominant, None)
            display = overlay_pil_on_cv2(display, emoji, ox, oy, ow, oh)
            text = f"{dominant} ({int(conf * 100)}%)"
            cv2.putText(display, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        FRAME_WINDOW.image(display, channels="BGR")

    cap.release()

st.markdown('<div class="footer">Made with ❤️ by Prabhant Tiwari</div>', unsafe_allow_html=True)
