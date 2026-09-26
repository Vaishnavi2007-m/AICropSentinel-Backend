import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd

# Live camera
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        WebRtcMode
    )
    import av
    CAMERA_ENABLED = True
except ImportError:
    CAMERA_ENABLED = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AICropSentinel",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f6f8f7;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e6e4;
    }

    /* Header */
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #183a2a;
        margin-bottom: 2px;
    }

    .main-subtitle {
        color: #65736b;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* Cards */
    .dashboard-card {
        background: #ffffff;
        border: 1px solid #e2e7e4;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .card-title {
        font-size: 17px;
        font-weight: 650;
        color: #24352b;
        margin-bottom: 12px;
    }

    .metric-label {
        color: #69776f;
        font-size: 13px;
    }

    .metric-value {
        color: #1e3026;
        font-size: 25px;
        font-weight: 700;
    }

    .healthy-box {
        background: #edf7f0;
        border: 1px solid #cce5d3;
        border-radius: 10px;
        padding: 14px;
    }

    .warning-box {
        background: #fff7e8;
        border: 1px solid #f1d79b;
        border-radius: 10px;
        padding: 14px;
    }

    .danger-box {
        background: #fff0f0;
        border: 1px solid #edc7c7;
        border-radius: 10px;
        padding: 14px;
    }

    .status-online {
        color: #18794e;
        font-weight: 650;
    }

    .status-offline {
        color: #b42318;
        font-weight: 650;
    }

    .small-text {
        font-size: 12px;
        color: #707b75;
    }

    /* Remove excessive Streamlit top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LANGUAGE DATA
# ============================================================

LANG = {

    "English": {
        "dashboard": "Dashboard",
        "live_camera": "Live Camera",
        "ai_analysis": "AI Analysis",
        "farm_status": "Farm Status",
        "rover_status": "Rover Status",
        "battery": "Battery",
        "soil": "Soil Moisture",
        "plants": "Plants Scanned",
        "healthy": "Healthy",
        "affected": "Affected",
        "disease": "Disease",
        "confidence": "Confidence",
        "treatment": "Treatment",
        "pending": "Pending",
        "completed": "Completed",
        "farm_map": "Farm Map",
        "history": "Detection History",
        "alerts": "Alerts",
        "voice": "Voice Alert",
        "language": "Language",
        "model": "ML Model",
        "ready": "Ready",
        "offline": "Offline",
        "online": "Online",
        "targeted_spray": "Targeted Spray",
        "start_spray": "Start Targeted Spray",
        "plant_id": "Plant ID",
        "crop": "Crop",
        "status": "Status",
        "system": "System Status"
    },

    "Tamil": {
        "dashboard": "Dashboard",
        "live_camera": "Live Camera",
        "ai_analysis": "AI Analysis",
        "farm_status": "Vayal Nilai",
        "rover_status": "Rover Nilai",
        "battery": "Battery",
        "soil": "Mann eerappatham",
        "plants": "Paarkkappatta Payirgal",
        "healthy": "Nalam",
        "affected": "Paathippu",
        "disease": "Noi",
        "confidence": "Nambikkai",
        "treatment": "Sigichai",
        "pending": "Kaathiruppu",
        "completed": "Mudinthathu",
        "farm_map": "Vayal Varaipadham",
        "history": "Detection Varalaru",
        "alerts": "Alertgal",
        "voice": "Kural Alert",
        "language": "Mozhi",
        "model": "ML Model",
        "ready": "Ready",
        "offline": "Offline",
        "online": "Online",
        "targeted_spray": "Thevaiyana Spray",
        "start_spray": "Targeted Spray Start",
        "plant_id": "Payir ID",
        "crop": "Payir",
        "status": "Nilai",
        "system": "System Nilai"
    },

    "Malayalam": {
        "dashboard": "Dashboard",
        "live_camera": "Live Camera",
        "ai_analysis": "AI Analysis",
        "farm_status": "Farm Status",
        "rover_status": "Rover Status",
        "battery": "Battery",
        "soil": "Manninte Eerppam",
        "plants": "Scan Cheytha Sasyangal",
        "healthy": "Arogyam",
        "affected": "Badha",
        "disease": "Rogam",
        "confidence": "Vishwasam",
        "treatment": "Chikitsa",
        "pending": "Pending",
        "completed": "Poorthiyayi",
        "farm_map": "Farm Map",
        "history": "Detection History",
        "alerts": "Alerts",
        "voice": "Voice Alert",
        "language": "Bhasha",
        "model": "ML Model",
        "ready": "Ready",
        "offline": "Offline",
        "online": "Online",
        "targeted_spray": "Targeted Spray",
        "start_spray": "Targeted Spray Start",
        "plant_id": "Plant ID",
        "crop": "Krishi",
        "status": "Nilavaram",
        "system": "System Status"
    },

    "Hindi": {
        "dashboard": "Dashboard",
        "live_camera": "Live Camera",
        "ai_analysis": "AI Vishleshan",
        "farm_status": "Khet Ki Sthiti",
        "rover_status": "Rover Sthiti",
        "battery": "Battery",
        "soil": "Mitti Ki Nami",
        "plants": "Scan Kiye Gaye Paudhe",
        "healthy": "Swasth",
        "affected": "Prabhavit",
        "disease": "Rog",
        "confidence": "Vishwas",
        "treatment": "Upchar",
        "pending": "Lambit",
        "completed": "Poorn",
        "farm_map": "Khet Map",
        "history": "Detection Itihas",
        "alerts": "Alerts",
        "voice": "Voice Alert",
        "language": "Bhasha",
        "model": "ML Model",
        "ready": "Ready",
        "offline": "Offline",
        "online": "Online",
        "targeted_spray": "Targeted Spray",
        "start_spray": "Targeted Spray Start",
        "plant_id": "Paudha ID",
        "crop": "Fasal",
        "status": "Sthiti",
        "system": "System Sthiti"
    },

    "Telugu": {
        "dashboard": "Dashboard",
        "live_camera": "Live Camera",
        "ai_analysis": "AI Vishleshana",
        "farm_status": "Pantala Sthithi",
        "rover_status": "Rover Sthithi",
        "battery": "Battery",
        "soil": "Matti Thadi",
        "plants": "Scan Chesina Pantalu",
        "healthy": "Arogyam",
        "affected": "Prabhavitham",
        "disease": "Vyadhi",
        "confidence": "Nammakam",
        "treatment": "Chikitsa",
        "pending": "Pending",
        "completed": "Poorthi",
        "farm_map": "Farm Map",
        "history": "Detection Charitra",
        "alerts": "Alerts",
        "voice": "Voice Alert",
        "language": "Bhasha",
        "model": "ML Model",
        "ready": "Ready",
        "offline": "Offline",
        "online": "Online",
        "targeted_spray": "Targeted Spray",
        "start_spray": "Targeted Spray Start",
        "plant_id": "Plant ID",
        "crop": "Pant",
        "status": "Sthithi",
        "system": "System Sthithi"
    },

    "Kannada": {
        "dashboard": "Dashboard",
        "live_camera": "Live Camera",
        "ai_analysis": "AI Vishleshane",
        "farm_status": "Bele Sthiti",
        "rover_status": "Rover Sthiti",
        "battery": "Battery",
        "soil": "Mannina Teva",
        "plants": "Scan Madida Belegalu",
        "healthy": "Arogya",
        "affected": "Prabhavita",
        "disease": "Roga",
        "confidence": "Vishwasa",
        "treatment": "Chikitse",
        "pending": "Pending",
        "completed": "Poorthi",
        "farm_map": "Farm Map",
        "history": "Detection Itihasa",
        "alerts": "Alerts",
        "voice": "Voice Alert",
        "language": "Bhashe",
        "model": "ML Model",
        "ready": "Ready",
        "offline": "Offline",
        "online": "Online",
        "targeted_spray": "Targeted Spray",
        "start_spray": "Targeted Spray Start",
        "plant_id": "Bele ID",
        "crop": "Bele",
        "status": "Sthiti",
        "system": "System Sthiti"
    }
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🌱 AICropSentinel")

language = st.sidebar.selectbox(
    "Language",
    list(LANG.keys())
)

T = LANG[language]

st.sidebar.markdown("---")

st.sidebar.markdown("### System")

if "system_online" not in st.session_state:
    st.session_state.system_online = True

if st.session_state.system_online:
    st.sidebar.success("● " + T["online"])
else:
    st.sidebar.warning("● " + T["offline"])

if st.sidebar.button("Toggle Network"):
    st.session_state.system_online = not st.session_state.system_online
    st.rerun()

st.sidebar.markdown("---")

if MODEL_STATUS if "MODEL_STATUS" in globals() else False:
    st.sidebar.success("ML Model: " + T["ready"])
else:
    st.sidebar.error("ML Model: Not Loaded")


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATH = "plant_disease_model.h5"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    MODEL_STATUS = True
    MODEL_ERROR = ""
except Exception as e:
    model = None
    MODEL_STATUS = False
    MODEL_ERROR = str(e)


# ============================================================
# IMPORTANT:
# REPLACE WITH YOUR ACTUAL 10 CLASS ORDER
# ============================================================

CLASS_NAMES = [
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4",
    "Class 5",
    "Class 6",
    "Class 7",
    "Class 8",
    "Class 9",
    "Class 10"
]


# ============================================================
# ML PREDICTION
# ============================================================

def predict_image(image):

    if model is None:
        return "Model unavailable", 0.0

    image = image.convert("RGB")
    image = image.resize((128, 128))

    array = np.array(image).astype("float32") / 255.0

    array = np.expand_dims(array, axis=0)

    prediction = model.predict(
        array,
        verbose=0
    )

    class_index = int(
        np.argmax(prediction[0])
    )

    confidence = float(
        prediction[0][class_index] * 100
    )

    disease = CLASS_NAMES[class_index]

    return disease, confidence


# ============================================================
# SESSION STATE
# ============================================================

if "last_disease" not in st.session_state:
    st.session_state.last_disease = "Waiting for scan"

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌱 AICropSentinel</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-subtitle">
    AI-driven crop health monitoring • Precision treatment •
    Digital farm monitoring
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP STATUS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="metric-label">🚜 {T["rover_status"]}</div>
            <div class="metric-value">ACTIVE</div>
            <div class="small-text">ROVER-01</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="metric-label">🔋 {T["battery"]}</div>
            <div class="metric-value">86%</div>
            <div class="small-text">Normal</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="metric-label">💧 {T["soil"]}</div>
            <div class="metric-value">62%</div>
            <div class="small-text">Moderate</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="metric-label">🌱 {T["plants"]}</div>
            <div class="metric-value">24</div>
            <div class="small-text">Today</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN CAMERA + AI RESULT
# ============================================================

camera_col, result_col = st.columns([1.5, 1])


# ============================================================
# LIVE CAMERA
# ============================================================

with camera_col:

    st.markdown(
        f'<div class="card-title">📷 {T["live_camera"]}</div>',
        unsafe_allow_html=True
    )

    if CAMERA_ENABLED:

        class CameraProcessor(VideoProcessorBase):

            def recv(self, frame):

                img = frame.to_ndarray(
                    format="bgr24"
                )

                return av.VideoFrame.from_ndarray(
                    img,
                    format="bgr24"
                )

        webrtc_streamer(
            key="aicrop-live-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=CameraProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

    else:

        st.warning(
            "Live camera module is not available."
        )


# ============================================================
# AI ANALYSIS
# ============================================================

with result_col:

    st.markdown(
        f'<div class="card-title">🤖 {T["ai_analysis"]}</div>',
        unsafe_allow_html=True
    )

    disease = st.session_state.last_disease
    confidence = st.session_state.last_confidence

    if confidence >= 70:

        st.markdown(
            f"""
            <div class="danger-box">
                <b>{T["disease"]}</b><br>
                <strong>{disease}</strong><br><br>
                {T["confidence"]}: {confidence:.2f}%<br>
                {T["status"]}: {T["affected"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif confidence > 0:

        st.markdown(
            f"""
            <div class="healthy-box">
                <b>{T["disease"]}</b><br>
                <strong>{disease}</strong><br><br>
                {T["confidence"]}: {confidence:.2f}%<br>
                {T["status"]}: {T["healthy"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="dashboard-card">
                Camera analysis waiting...
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DEVELOPER / TEST IMAGE
# ============================================================

st.markdown("---")

st.markdown(
    f"### 🧪 {T['ai_analysis']} — Test Image"
)

test_image = st.file_uploader(
    "Select a plant image for testing",
    type=["jpg", "jpeg", "png"]
)

if test_image:

    image = Image.open(test_image)

    st.image(
        image,
        caption="Test Image",
        width=450
    )

    if st.button(
        "Run AI Detection",
        type="primary"
    ):

        if MODEL_STATUS:

            with st.spinner(
                "AI model analysing image..."
            ):

                disease, confidence = predict_image(
                    image
                )

            st.session_state.last_disease = disease
            st.session_state.last_confidence = confidence
            st.session_state.last_image = image

            st.session_state.history.append({
                "Time": datetime.now().strftime(
                    "%H:%M:%S"
                ),
                "Plant ID": f"P{len(st.session_state.history)+1:03d}",
                "Crop": "Tomato",
                "Disease": disease,
                "Confidence": f"{confidence:.2f}%",
                "Treatment": (
                    "Targeted Spray"
                    if confidence >= 70
                    else "None"
                )
            })

            st.rerun()

        else:

            st.error(
                "ML model could not be loaded."
            )


# ============================================================
# TREATMENT
# ============================================================

st.markdown("---")

st.markdown(
    f"### 💧 {T['treatment']}"
)

t1, t2, t3, t4 = st.columns(4)

with t1:
    st.markdown(
        f"""
        <div class="dashboard-card">
        <div class="metric-label">{T["plant_id"]}</div>
        <div class="metric-value">P024</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with t2:
    st.markdown(
        f"""
        <div class="dashboard-card">
        <div class="metric-label">{T["crop"]}</div>
        <div class="metric-value">Tomato</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with t3:
    st.markdown(
        f"""
        <div class="dashboard-card">
        <div class="metric-label">{T["disease"]}</div>
        <div class="metric-value">
        {st.session_state.last_disease}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with t4:

    if st.session_state.last_confidence >= 70:

        if st.button(
            "🚜 " + T["start_spray"],
            use_container_width=True
        ):
            st.success(
                "Targeted spray command recorded."
            )

    else:

        st.success(
            "No treatment required"
        )


# ============================================================
# FARM OVERVIEW
# ============================================================

st.markdown("---")

st.markdown(
    f"### 🌾 {T['farm_status']}"
)

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.metric(
        "Total Plants",
        "120"
    )

with f2:
    st.metric(
        T["healthy"],
        "96"
    )

with f3:
    st.metric(
        T["affected"],
        "24"
    )

with f4:
    st.metric(
        T["treatment"],
        "18"
    )


# ============================================================
# FARM MAP
# ============================================================

st.markdown("---")

st.markdown(
    f"### 🗺️ {T['farm_map']}"
)

map_data = pd.DataFrame({
    "lat": [
        11.0168,
        11.0171,
        11.0164,
        11.0173,
        11.0167
    ],
    "lon": [
        76.9558,
        76.9562,
        76.9554,
        76.9565,
        76.9560
    ]
})

st.map(
    map_data,
    zoom=17
)


# ============================================================
# DETECTION HISTORY
# ============================================================

st.markdown("---")

st.markdown(
    f"### 📋 {T['history']}"
)

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    demo_history = pd.DataFrame({
        "Time": ["09:42", "09:48", "09:55"],
        "Plant ID": ["P021", "P022", "P023"],
        "Crop": ["Tomato", "Tomato", "Tomato"],
        "Disease": [
            "Healthy",
            "Disease detected",
            "Healthy"
        ],
        "Confidence": [
            "97%",
            "91%",
            "98%"
        ],
        "Treatment": [
            "None",
            "Completed",
            "None"
        ]
    })

    st.dataframe(
        demo_history,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ALERTS
# ============================================================

st.markdown("---")

st.markdown(
    f"### ⚠️ {T['alerts']}"
)

if st.session_state.last_confidence >= 70:

    st.error(
        f"Plant P024 requires attention. "
        f"{st.session_state.last_disease} detected "
        f"with {st.session_state.last_confidence:.1f}% confidence."
    )

else:

    st.success(
        "No critical disease alert currently active."
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown("---")

st.markdown(
    f"### ⚙️ {T['system']}"
)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.success("AI Model — Ready")

with s2:
    st.success("Camera — Ready")

with s3:
    st.success("Local Storage — Ready")

with s4:

    if st.session_state.system_online:
        st.success("Network — Online")
    else:
        st.warning("Network — Offline")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#7a847e;font-size:12px;">
    AICropSentinel • PhytoGuard<br>
    AI-driven autonomous agricultural rover platform
    </div>
    """,
    unsafe_allow_html=True
)