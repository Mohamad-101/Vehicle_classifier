import io
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="VisionAI · Vehicle Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --bg: #0f1320;
        --card: #161b2c;
        --border: #262d44;
        --muted: #8a93ab;
        --fg: #eef1f8;
        --primary: #38d6ff;
        --accent:  #c46bff;
        --success: #4ade80;
        --danger:  #ff6b6b;
      }
      .stApp { background:
        radial-gradient(ellipse at top, rgba(56,214,255,.10), transparent 55%),
        radial-gradient(ellipse at bottom right, rgba(196,107,255,.10), transparent 55%),
        var(--bg);
        color: var(--fg);
      }
      .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px;}
      h1, h2, h3, h4 { color: var(--fg); letter-spacing: -0.01em; }
      .va-eyebrow {
        display:inline-flex; align-items:center; gap:.5rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size:.72rem; text-transform:uppercase; letter-spacing:.18em;
        color: var(--muted);
        padding:.35rem .7rem; border:1px solid var(--border);
        border-radius:999px; background: rgba(255,255,255,.02);
      }
      .va-title { font-size: 2.6rem; font-weight: 800; line-height:1.05; margin:.6rem 0 .4rem; }
      .va-grad { background: linear-gradient(135deg, var(--primary), var(--accent));
                 -webkit-background-clip:text; background-clip:text; color: transparent; }
      .va-sub { color: var(--muted); font-size:1.02rem; max-width: 720px; }
      .va-card {
        background: var(--card); border:1px solid var(--border);
        border-radius: 14px; padding: 1.1rem 1.2rem;
      }
      .va-stat-k {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        color: var(--muted); font-size:.68rem; letter-spacing:.18em;
        text-transform:uppercase; margin-bottom:.25rem;
      }
      .va-stat-v { font-weight:700; color: var(--fg); font-size:1.05rem; }
      .va-pill {
        display:inline-flex; align-items:center; gap:.4rem;
        padding:.3rem .7rem; border-radius:999px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:700;
      }
      .va-pill.ok { background: rgba(74,222,128,.12); color: var(--success); }
      .va-pill.no { background: rgba(255,107,107,.12); color: var(--danger); }
      .va-step {
        display:inline-flex; align-items:center; gap:.55rem;
        font-family: ui-monospace, monospace; font-size:.7rem;
        text-transform:uppercase; letter-spacing:.18em; color: var(--muted);
      }
      .va-step .num {
        width:1.4rem; height:1.4rem; border-radius:.4rem;
        background: rgba(56,214,255,.15); color: var(--primary);
        display:grid; place-items:center; font-weight:800;
      }
      .va-step.alt .num { background: rgba(196,107,255,.15); color: var(--accent); }
      .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--accent));
      }
      section[data-testid="stSidebar"] { background: #11162a; border-right:1px solid var(--border); }
      .va-dot { width:.55rem; height:.55rem; border-radius:50%; background: var(--success);
                box-shadow: 0 0 0 0 rgba(74,222,128,.6); animation: vapulse 2s infinite; display:inline-block;}
      @keyframes vapulse {
        0%   { box-shadow: 0 0 0 0 rgba(74,222,128,.6); }
        70%  { box-shadow: 0 0 0 10px rgba(74,222,128,0); }
        100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); }
      }
      [data-testid="stFileUploader"] section {
        background: var(--card); border: 2px dashed var(--border); border-radius: 14px;
      }
      [data-testid="stFileUploader"] section:hover { border-color: var(--primary); }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_ai_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

def analyze_image(image: Image.Image, model):
    img = image.convert("RGB").resize((224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    preds = model.predict(arr, verbose=0)
    return tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

VEHICLE_KEYWORDS = [
    "car", "truck", "bus", "van", "jeep", "wagon", "cab", "ambulance",
    "scooter", "motorcycle", "moped", "limousine", "sports_car",
    "minivan", "convertible", "pickup", "tow_truck", "trailer", "tractor",
]

def is_vehicle(predictions):
    for _id, label, conf in predictions:
        if any(k in label.lower() for k in VEHICLE_KEYWORDS):
            return True, label.replace("_", " ").title(), float(conf)
    top = predictions[0]
    return False, top[1].replace("_", " ").title(), float(top[2])

with st.sidebar:
    st.markdown("### ⚙️ System")
    st.markdown('<span class="va-dot"></span> &nbsp;Neural network online',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div class="va-stat-k">Engine</div><div class="va-stat-v">MobileNetV2</div>'
        '<br><div class="va-stat-k">Weights</div><div class="va-stat-v">ImageNet · 1,000 classes</div>'
        '<br><div class="va-stat-k">Input</div><div class="va-stat-v">224 × 224 RGB</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    with st.expander("📖 User guide", expanded=True):
        st.write(
            "1. Upload a clear photo (JPG / PNG).\n"
            "2. The model classifies it into 3 top categories.\n"
            "3. We check those against a vehicle keyword list."
        )

with st.spinner("Booting neural network…"):
    model = load_ai_model()

st.markdown('<span class="va-eyebrow">✦ Real-time computer vision · MobileNetV2</span>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="va-title">Vehicle detection,<br/>'
    '<span class="va-grad">powered by deep learning.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="va-sub">Drop any image and a convolutional neural network classifies '
    'it across 1,000 ImageNet categories — then decides whether it shows a vehicle.</div>',
    unsafe_allow_html=True,
)

st.write("")
col1, col2 = st.columns([1, 1.15], gap="large")

with col1:
    st.markdown('<div class="va-step"><span class="num">1</span> Input</div>',
                unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader(
        "Drop an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    image = None
    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, use_container_width=True, caption="Source image")

with col2:
    st.markdown('<div class="va-step alt"><span class="num">2</span> Analysis</div>',
                unsafe_allow_html=True)
    st.write("")

    if image is None:
        st.markdown(
            '<div class="va-card" style="text-align:center; padding:2.2rem 1.2rem;">'
            '<div style="font-size:1.05rem; font-weight:600; margin-bottom:.3rem;">Awaiting input</div>'
            '<div style="color: var(--muted); font-size:.92rem;">Upload an image to run inference.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Running forward pass…"):
            t0 = time.perf_counter()
            preds = analyze_image(image, model)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            detected, top_label, confidence = is_vehicle(preds)

        pill = ('<span class="va-pill ok">✓ Vehicle detected</span>' if detected
                else '<span class="va-pill no">✕ No vehicle</span>')

        st.markdown(
            f'''
            <div class="va-card">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.9rem;">
                {pill}
                <span style="font-family:ui-monospace, monospace; font-size:.78rem; color: var(--muted);">⚡ {elapsed_ms} ms</span>
              </div>
              <div class="va-stat-k">Primary classification</div>
              <div style="font-size:1.55rem; font-weight:800; margin:.15rem 0 1rem;">{top_label}</div>
              <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span class="va-stat-k">Confidence</span>
                <span class="va-grad" style="font-family:ui-monospace, monospace; font-size:1.4rem; font-weight:800;">{confidence*100:.1f}%</span>
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.progress(int(confidence * 100))

        st.write("")
        with st.expander("🔍 Top-3 telemetry", expanded=True):
            for _id, label, conf in preds:
                name = label.replace("_", " ").title()
                pct = float(conf) * 100
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; '
                    f'font-size:.92rem; margin-bottom:.25rem;">'
                    f'<span>{name}</span>'
                    f'<span style="font-family:ui-monospace, monospace; color: var(--muted);">{pct:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(int(pct))

st.write("")
st.write("")
c1, c2, c3, c4 = st.columns(4)
specs = [
    ("Engine", "MobileNetV2"),
    ("Classes", "1,000"),
    ("Runtime", "TensorFlow / Keras"),
    ("Latency", f"{elapsed_ms} ms" if image is not None else "—"),
]
for col, (k, v) in zip((c1, c2, c3, c4), specs):
    with col:
        st.markdown(
            f'<div class="va-card"><div class="va-stat-k">{k}</div>'
            f'<div class="va-stat-v">{v}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
st.markdown(
    '<div style="text-align:center; color: var(--muted); font-family:ui-monospace, monospace; '
    'font-size:.75rem; margin-top:1.2rem;">Weights: ImageNet · License: Apache 2.0</div>',
    unsafe_allow_html=True,
)