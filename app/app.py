import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import io

# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "models/vehicle_mobilenetv2_final.keras"
IMG_SIZE = (224, 224)

# Urutan HARUS sama dengan urutan label saat training
CLASS_NAMES = [
    "SUV",
    "bus",
    "convertible",
    "coupes",
    "hatchback",
    "pickup",
    "sedan",
    "station_wagon",
    "trucks",
    "van",
]

# nama tampilan, emoji, deskripsi singkat
CLASS_INFO = {
    "SUV":           ("SUV",           "🚙", "Tall body, high ground clearance, often 4WD."),
    "bus":           ("Bus",           "🚌", "Large passenger vehicle with many seats."),
    "convertible":   ("Convertible",   "🏎️", "Car with a retractable or removable roof."),
    "coupes":        ("Coupe",         "🚘", "Two-door car with a sloping, sporty roofline."),
    "hatchback":     ("Hatchback",     "🚗", "Compact body with a rear lift-up door."),
    "pickup":        ("Pickup",        "🛻", "Open cargo bed behind the cabin."),
    "sedan":         ("Sedan",         "🚖", "Four doors with a separate trunk."),
    "station_wagon": ("Station wagon", "🚐", "Extended roofline with a large rear cargo area."),
    "trucks":        ("Truck",         "🚚", "Heavy vehicle built to haul freight."),
    "van":           ("Van",           "🚐", "Boxy body for people or cargo."),
}


# =========================
# PAGE
# =========================

st.set_page_config(
    page_title="Vehicle Type Classifier",
    page_icon="🚗",
    layout="centered",
)


# =========================
# THEME (light / dark)
# =========================

THEMES = {
    "light": {
        "ink": "#0F1115", "muted": "#646B78", "line": "#E3E5EA",
        "paper": "#F5F6F8", "card": "#FFFFFF", "accent": "#F4511E",
        "accent-soft": "#FFE9E2", "track": "#E9EBF0", "dash": "#C3C8D2",
        "glow": "rgba(244,81,30,.12)", "shadow": "0 1px 2px rgba(15,17,21,.04), 0 8px 24px rgba(15,17,21,.06)",
    },
    "dark": {
        "ink": "#F2F4F8", "muted": "#8A92A3", "line": "#232833",
        "paper": "#0A0C10", "card": "#12151C", "accent": "#FF7A4D",
        "accent-soft": "#2B1812", "track": "#1E2330", "dash": "#3A4152",
        "glow": "rgba(255,122,77,.18)", "shadow": "0 1px 2px rgba(0,0,0,.4), 0 12px 32px rgba(0,0,0,.35)",
    },
}

st.session_state.setdefault("dark_mode", False)
is_dark = st.session_state["dark_mode"]
T = THEMES["dark" if is_dark else "light"]

theme_vars = "".join(f"--{k}: {v};" for k, v in T.items())
st.markdown(
    f"""<style>
:root {{ {theme_vars} color-scheme: {"dark" if is_dark else "light"}; }}
</style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }
.stApp {
    background:
        radial-gradient(900px 380px at 85% -80px, var(--glow), transparent 70%),
        var(--paper);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 860px; padding-top: 1.4rem; padding-bottom: 4rem; }

/* Paksa warna teks mengikuti tema aplikasi */
.stApp p, .stApp span, .stApp label, .stApp li,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"], [data-testid="stSpinner"] { color: var(--ink); }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: var(--ink); }
hr { border-color: var(--line) !important; }

/* Navbar */
.brand { display: flex; align-items: center; gap: .7rem; }
.brand-logo {
    width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
    background: var(--accent); font-size: 1.2rem;
}
.brand-name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; letter-spacing: -.01em; }
.brand-name small { display: block; font-family: 'Inter', sans-serif; font-weight: 500; font-size: .72rem; color: var(--muted); letter-spacing: 0; }

/* Hero */
.hero { padding: 2.4rem 0 1.4rem; }
.badge {
    display: inline-flex; align-items: center; gap: .45rem; font-size: .78rem; font-weight: 600;
    color: var(--accent); background: var(--accent-soft); padding: .3rem .75rem; border-radius: 999px;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 3rem; line-height: 1.05;
    letter-spacing: -.03em; margin: 1rem 0 .8rem; color: var(--ink);
}
.hero h1 em { font-style: normal; color: var(--accent); }
.hero p { color: var(--muted) !important; font-size: 1.05rem; max-width: 32rem; margin: 0; }

/* Tabs */
div[data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); background: transparent; gap: .4rem; }
button[data-baseweb="tab"] { background: transparent !important; font-weight: 600; }
button[data-baseweb="tab"] p, button[data-baseweb="tab"] span { color: var(--muted) !important; font-weight: 600; }
button[data-baseweb="tab"]:hover p { color: var(--ink) !important; }
button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span { color: var(--accent) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
div[data-baseweb="tab-border"] { background-color: var(--line) !important; }

/* Uploader & kamera */
[data-testid="stFileUploaderDropzone"] {
    background: var(--card); border: 1.5px dashed var(--dash); border-radius: 20px;
    padding: 2rem 1.6rem; box-shadow: var(--shadow);
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); }
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stCameraInput"] button { background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stCameraInput"] button *,
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }

/* Toggle */
[data-testid="stToggle"] label p { font-weight: 600; font-size: .9rem; }

/* Gambar */
[data-testid="stImage"] img { border-radius: 20px; border: 1px solid var(--line); box-shadow: var(--shadow); }
[data-testid="stImageCaption"] { color: var(--muted); font-size: .85rem; }

/* Panel hasil */
.panel {
    background: var(--card); border: 1px solid var(--line); border-radius: 20px;
    padding: 1.5rem; box-shadow: var(--shadow);
}
.kicker { font-size: .74rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.result-name {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.6rem; line-height: 1.05;
    letter-spacing: -.03em; margin: .5rem 0 .4rem; display: flex; align-items: center; gap: .6rem;
}
.result-desc { color: var(--muted); font-size: .95rem; margin-bottom: 1.3rem; }
.conf-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: .5rem; }
.conf-head b { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; }
.segments { display: grid; grid-template-columns: repeat(20, 1fr); gap: 3px; }
.seg { height: 14px; border-radius: 4px; background: var(--track); }
.seg.on { background: var(--accent); }

.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.2rem; }
.chip {
    font-size: .82rem; font-weight: 500; padding: .35rem .7rem; border-radius: 999px;
    border: 1px solid var(--line); color: var(--ink); background: var(--paper);
}
.chip b { color: var(--muted); font-weight: 600; margin-left: .3rem; }

/* Semua kelas */
.section-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.25rem; letter-spacing: -.01em; margin: 2rem 0 .8rem; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .6rem; }
.tile {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: .75rem .95rem;
}
.tile.top { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.tile-row { display: flex; justify-content: space-between; font-size: .9rem; font-weight: 600; }
.tile-row span:last-child { color: var(--muted); font-weight: 500; font-variant-numeric: tabular-nums; }
.tile.top .tile-row span:last-child { color: var(--accent); font-weight: 700; }
.bar { height: 5px; border-radius: 999px; background: var(--track); margin-top: .5rem; overflow: hidden; }
.fill { height: 100%; border-radius: 999px; background: var(--dash); }
.tile.top .fill { background: var(--accent); }

.note { color: var(--muted); font-size: .85rem; margin-top: 1.6rem; }

@media (max-width: 640px) {
    .hero h1 { font-size: 2.1rem; }
    .result-name { font-size: 2rem; }
    .grid { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: no-preference) { .fill { transition: width .5s ease; } }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model tidak bisa dimuat. Pastikan file `{MODEL_PATH}` ada.")
    st.caption(f"Detail error: {e}")
    st.stop()


# =========================
# PREDICTION
# =========================

@st.cache_data(show_spinner=False)
def predict_from_bytes(data: bytes):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    resized = image.resize(IMG_SIZE)
    arr = np.expand_dims(np.array(resized), axis=0)
    return model.predict(arr, verbose=0)[0]


def render_result(probs):
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    key = CLASS_NAMES[top]
    name, emoji, desc = CLASS_INFO[key]
    conf = float(probs[top]) * 100

    on = int(round(conf / 5))
    segs = "".join(f'<div class="seg{" on" if i < on else ""}"></div>' for i in range(20))

    chips = "".join(
        f'<span class="chip">{CLASS_INFO[CLASS_NAMES[i]][0]}<b>{float(probs[i]) * 100:.1f}%</b></span>'
        for i in order[1:3]
    )

    st.markdown(
        f"""<div class="panel">
<div class="kicker">Predicted body type</div>
<div class="result-name"><span>{emoji}</span>{name}</div>
<div class="result-desc">{desc}</div>
<div class="conf-head"><span class="kicker">Confidence</span><b>{conf:.1f}%</b></div>
<div class="segments">{segs}</div>
<div class="chips"><span class="kicker" style="align-self:center">Also possible</span>{chips}</div>
</div>""",
        unsafe_allow_html=True,
    )
    return conf, order


def render_all(probs, order):
    tiles = []
    for rank, i in enumerate(order):
        name = CLASS_INFO[CLASS_NAMES[i]][0]
        pct = float(probs[i]) * 100
        top = " top" if rank == 0 else ""
        tiles.append(
            f'<div class="tile{top}"><div class="tile-row"><span>{name}</span><span>{pct:.1f}%</span></div>'
            f'<div class="bar"><div class="fill" style="width:{pct:.2f}%"></div></div></div>'
        )
    st.markdown(
        '<div class="section-title">All classes</div><div class="grid">' + "".join(tiles) + "</div>",
        unsafe_allow_html=True,
    )


def analyze(data: bytes):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    probs = predict_from_bytes(data)

    col_img, col_res = st.columns([5, 6], gap="medium")
    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col_res:
        conf, order = render_result(probs)

    if conf < 50:
        st.warning(
            "Confidence is low. Try a clearer photo where the whole vehicle is visible "
            "from the side or a front three-quarter angle."
        )

    render_all(probs, order)


# =========================
# MAIN
# =========================

nav_l, nav_r = st.columns([3, 1], vertical_alignment="center")
with nav_l:
    st.markdown(
        """<div class="brand"><div class="brand-logo">🚗</div>
<div class="brand-name">VehicleLens<small>Body type classifier</small></div></div>""",
        unsafe_allow_html=True,
    )
with nav_r:
    st.toggle("🌙 Dark mode", key="dark_mode")

st.markdown(
    """<div class="hero">
<span class="badge">● 10 body types</span>
<h1>Mengenali setiap kendaraan<br>hanya <em>sekali lihat</em>.</h1>
<p>Unggah foto dan model tersebut akan mengklasifikasikan tipe bodi kendaraan, mulai dari SUV dan sedan hingga bus dan truk.</p>
</div>""",
    unsafe_allow_html=True,
)

tab_upload, tab_camera = st.tabs(["Upload image", "Use camera"])

image_bytes = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()

with tab_camera:
    camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
    if camera_file is not None:
        image_bytes = camera_file.getvalue()

if image_bytes is not None:
    st.divider()
    with st.spinner("Classifying..."):
        analyze(image_bytes)
else:
    st.markdown('<div class="note">No image yet. Upload a vehicle photo or use your camera to start.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="note">Predictions come from a MobileNetV2 model. Results may vary with angle, lighting, and image quality.</div>',
    unsafe_allow_html=True,
)