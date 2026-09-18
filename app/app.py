"""Streamlit Web Application for Handwritten Character Classifier (VisionX 2026 - IC-09)."""
import os
import io
import sys
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import CharacterPredictor, DEFAULT_CONFIDENCE_THRESHOLD
from src.model import CLASS_MAPPING

# -------------------------------------------------------------
# Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Character Classifier",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional hackathon presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .char-badge {
        font-size: 4.5rem;
        font-weight: 800;
        color: #2563EB;
        line-height: 1;
        text-align: center;
        margin: 0.5rem 0;
    }
    .uncertain-badge {
        font-size: 2.8rem;
        font-weight: 800;
        color: #D97706;
        line-height: 1.2;
        text-align: center;
        margin: 0.5rem 0;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Cached Model Loader
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained CNN model into memory...")
def get_cached_predictor() -> CharacterPredictor:
    """Load model once and cache the predictor instance across all Streamlit sessions."""
    model_path = os.path.join(PROJECT_ROOT, "models", "handwritten_character_model.keras")
    if not os.path.exists(model_path):
        st.error(f"Trained model not found at '{model_path}'. Please run training/train.py first.")
        st.stop()
    return CharacterPredictor.get_instance(model_path)

predictor = get_cached_predictor()

# -------------------------------------------------------------
# Sidebar: Settings & Model/Dataset Information
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("## ⚙️ Settings")
    
    threshold_pct = st.slider(
        "Confidence Threshold (%)",
        min_value=50,
        max_value=99,
        value=int(DEFAULT_CONFIDENCE_THRESHOLD * 100),
        step=1,
        help="Predictions with confidence below this threshold will be flagged as 'Uncertain' instead of forcing an incorrect classification."
    )
    confidence_threshold = threshold_pct / 100.0

    st.markdown("---")
    st.markdown("### 🧠 Model Information")
    st.markdown("""
    * **Architecture**: Convolutional Neural Network (CNN)
    * **Structure**: 3 Conv Blocks + BatchNorm + MaxPool + Dropout + Softmax
    * **Input Resolution**: 28 × 28 Grayscale
    * **Output Classes**: 26 Uppercase English Letters (A–Z)
    * **Measured Test Accuracy**: **93.36%**  
      *(Evaluated on untouched EMNIST Letters test set of 20,800 images)*
    * **Measured Test Loss**: **0.1997**
    """)

    st.markdown("---")
    st.markdown("### 📚 Dataset Information")
    st.markdown("""
    * **Dataset**: EMNIST Letters (NIST Special Database 19)
    * **Training Set**: 124,800 handwritten images
    * **Test Set**: 20,800 handwritten images
    * **Balance**: Exactly 800 test images per character
    """)

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="main-header">✍️ Handwritten Character Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered handwritten English character recognition (VisionX 2026 — IC-09)</div>', unsafe_allow_html=True)
st.info("Upload an image or draw a handwritten character to classify it into one of 26 English uppercase letters (A–Z). The system includes confidence scoring and low-confidence gating for OCR safety.")

# -------------------------------------------------------------
# Input Modes: Tabs
# -------------------------------------------------------------
tab_upload, tab_draw, tab_samples = st.tabs([
    "📁 Upload Image",
    "✏️ Interactive Drawing Pad",
    "🎯 Quick Demo Presets"
])

current_image = None
image_source_label = ""

# --- Tab 1: Upload Image ---
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a handwritten character image (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
        key="file_uploader"
    )
    if uploaded_file is not None:
        try:
            current_image = Image.open(uploaded_file)
            image_source_label = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Unable to read uploaded image file: {e}")

# --- Tab 2: Interactive Drawing Pad ---
with tab_draw:
    st.markdown("Draw a single English uppercase letter (**A–Z**) below using your mouse or trackpad:")
    
    canvas_html = """
    <div style="display: flex; flex-direction: column; align-items: center; background: #1E293B; border-radius: 12px; padding: 16px; width: 340px; margin: 0 auto;">
        <canvas id="paintCanvas" width="280" height="280" style="background: black; border-radius: 8px; cursor: crosshair; touch-action: none; border: 2px solid #475569;"></canvas>
        <div style="display: flex; gap: 10px; margin-top: 12px;">
            <button onclick="clearCanvas()" style="background: #EF4444; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 600; cursor: pointer;">Clear Canvas</button>
            <button onclick="downloadCanvas()" style="background: #2563EB; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 600; cursor: pointer;">Download for Upload</button>
        </div>
        <p style="color: #94A3B8; font-size: 12px; margin-top: 8px; text-align: center;">Draw with white ink on black canvas, then click Download and drop into Upload Image tab.</p>
    </div>
    <script>
        const canvas = document.getElementById('paintCanvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'white';

        let drawing = false;

        function startPos(e) {
            drawing = true;
            draw(e);
        }
        function endPos() {
            drawing = false;
            ctx.beginPath();
        }
        function draw(e) {
            if (!drawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX || (e.touches && e.touches[0].clientX)) - rect.left;
            const y = (e.clientY || (e.touches && e.touches[0].clientY)) - rect.top;
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        canvas.addEventListener('mousedown', startPos);
        canvas.addEventListener('mouseup', endPos);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startPos(e); });
        canvas.addEventListener('touchend', endPos);
        canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        function downloadCanvas() {
            const link = document.createElement('a');
            link.download = 'handwritten_drawn_letter.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }
    </script>
    """
    components.html(canvas_html, height=390)
    st.caption("Tip: After sketching your character, click **Download for Upload** and drop the saved PNG into the **Upload Image** tab.")

# --- Tab 3: Quick Demo Presets ---
with tab_samples:
    st.markdown("Test the classifier instantly using pre-generated verified handwritten samples:")
    sample_dir = os.path.join(PROJECT_ROOT, "data", "test_samples")
    
    col_s1, col_s2, col_s3, col_s4, col_s5, col_s6 = st.columns(6)
    
    with col_s1:
        if st.button("Letter 'A' (EMNIST)", use_container_width=True):
            p = os.path.join(sample_dir, "emnist_A.png")
            if os.path.exists(p):
                current_image = Image.open(p)
                image_source_label = "Preset: Real EMNIST Test Sample 'A'"

    with col_s2:
        if st.button("Letter 'B' (EMNIST)", use_container_width=True):
            p = os.path.join(sample_dir, "emnist_B.png")
            if os.path.exists(p):
                current_image = Image.open(p)
                image_source_label = "Preset: Real EMNIST Test Sample 'B'"

    with col_s3:
        if st.button("Letter 'M' (EMNIST)", use_container_width=True):
            p = os.path.join(sample_dir, "emnist_M.png")
            if os.path.exists(p):
                current_image = Image.open(p)
                image_source_label = "Preset: Real EMNIST Test Sample 'M'"

    with col_s4:
        if st.button("Letter 'Z' (EMNIST)", use_container_width=True):
            p = os.path.join(sample_dir, "emnist_Z.png")
            if os.path.exists(p):
                current_image = Image.open(p)
                image_source_label = "Preset: Real EMNIST Test Sample 'Z'"

    with col_s5:
        if st.button("Ambiguous (Low Conf)", use_container_width=True):
            p = os.path.join(sample_dir, "emnist_ambiguous_low_conf.png")
            if os.path.exists(p):
                current_image = Image.open(p)
                image_source_label = "Preset: Real Ambiguous Sample (Low Confidence)"

    with col_s6:
        if st.button("Blank Image (Invalid)", use_container_width=True):
            p = os.path.join(sample_dir, "blank_sample.png")
            if os.path.exists(p):
                current_image = Image.open(p)
                image_source_label = "Preset: Blank Image"

# Store active image in session state so switching tabs preserves it
if current_image is not None:
    st.session_state["active_image"] = current_image
    st.session_state["active_label"] = image_source_label

active_image = st.session_state.get("active_image", None)
active_label = st.session_state.get("active_label", "")

# -------------------------------------------------------------
# Previews & Prediction Flow
# -------------------------------------------------------------
st.markdown("---")

if active_image is not None:
    col_input, col_processed = st.columns(2)
    
    with col_input:
        st.markdown(f"**Input Image** *({active_label})*")
        st.image(active_image, width=220)

    # Run quick preview preprocessing for visualization without full inference yet
    from src.preprocessing import preprocess_user_image
    _, is_valid_preview, _, preview_28x28 = preprocess_user_image(active_image)

    with col_processed:
        st.markdown("**Model Input — 28 × 28 Grayscale** *(Centered & Normalized)*")
        st.image(preview_28x28, width=220, clamp=True)
        if not is_valid_preview:
            st.caption("⚠️ No distinct character strokes detected in input.")

    st.markdown("")
    classify_btn = st.button("🔍 Classify Character", type="primary", use_container_width=True)

    if classify_btn:
        with st.spinner("Classifying character through CNN pipeline..."):
            result = predictor.predict(active_image, confidence_threshold=confidence_threshold)

        st.markdown("### 📊 Classification Result")

        # Case 1: Invalid / Blank input
        if result["status"] == "invalid":
            st.error(f"❌ **Invalid Input:** {result['message']}")
            st.info("Please provide an image or sketch with a clear, visible handwritten character stroke.")

        # Case 2: Uncertain (Low Confidence)
        elif result["status"] == "uncertain":
            st.warning("⚠️ **Prediction: Uncertain**")
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.markdown(f'<div class="uncertain-badge">?</div>', unsafe_allow_html=True)
                st.markdown(f"<center><b>Candidate: '{result['raw_prediction']}'</b> ({result['confidence_percent']}%)</center>", unsafe_allow_html=True)
            with col_res2:
                st.markdown(f"**Status Message**: {result['message']}")
                st.progress(float(result["confidence"]))
                st.markdown(f"*Confidence ({result['confidence_percent']}%) is below the configured threshold of {threshold_pct}%.*")
            
            # Show Top-3 even for uncertain predictions to inspect candidates
            st.markdown("#### Top-3 Candidate Predictions:")
            for item in result["top_3"]:
                c1, c2, c3 = st.columns([1, 2, 7])
                c1.markdown(f"**Rank {item['rank']}**")
                c2.markdown(f"**Character: `{item['character']}`**")
                c3.progress(item["probability"], text=f"{item['percentage']}")

        # Case 3: Confident / Successful Classification
        else:
            st.success(f"✅ **Prediction Complete**")
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.markdown(f'<div class="char-badge">{result["prediction"]}</div>', unsafe_allow_html=True)
                st.markdown(f"<center><b>Predicted Character: '{result['prediction']}'</b></center>", unsafe_allow_html=True)
            with col_res2:
                st.markdown(f'<div class="metric-label">Confidence Score</div>', unsafe_allow_html=True)
                st.markdown(f"### {result['confidence_percent']}%")
                st.progress(float(result["confidence"]))
                st.markdown(f"*{result['message']}*")

            # Top-3 Breakdown
            st.markdown("#### 🏆 Top-3 Predictions:")
            for item in result["top_3"]:
                c1, c2, c3 = st.columns([1, 2, 7])
                c1.markdown(f"**Rank {item['rank']}**")
                c2.markdown(f"**Character: `{item['character']}`**")
                c3.progress(item["probability"], text=f"{item['percentage']}")

        # Collapsible Full 26-Class Probability Chart
        if result["status"] != "invalid":
            with st.expander("📈 View All 26 Classes Probability Distribution"):
                all_probs = result["all_probabilities"]
                chart_data = {CLASS_MAPPING[i]: all_probs[i] * 100 for i in range(26)}
                st.bar_chart(chart_data)
else:
    st.info("👆 Upload an image in the **Upload Image** tab or select a sample from **Quick Demo Presets** to begin.")
