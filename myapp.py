import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import time

# --- STYLING ---
st.set_page_config(page_title="Numberplate Scanner", page_icon="🚗", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f8f9fc;
    }
    .history-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_keras_model():
    from keras.models import load_model
    try:
        model = load_model("keras_model.h5", compile=False)
        with open("labels.txt", "r") as f:
            class_names = f.readlines()
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, class_names = load_keras_model()

# --- STATE INITIALIZATION ---
if "history" not in st.session_state:
    st.session_state.history = []

if "scan_trigger" not in st.session_state:
    st.session_state.scan_trigger = 0
    
if "last_processed_hash" not in st.session_state:
    st.session_state.last_processed_hash = None

def reset_scan():
    st.session_state.scan_trigger += 1
    st.session_state.last_processed_hash = None

def add_to_history(image, country, confidence):
    st.session_state.history.insert(0, {
        "image": image,
        "country": country,
        "confidence": confidence,
        "time": time.strftime("%H:%M:%S")
    })
    st.session_state.history = st.session_state.history[:5]

def process_image(img):
    if model is None or class_names is None:
        return None, 0.0
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name_raw = class_names[index].strip()
    
    # Extract just the label name if it has an index prefix like '0 ClassName'
    class_name = class_name_raw[2:] if len(class_name_raw) >= 2 and class_name_raw[1] == ' ' else class_name_raw
    confidence_score = float(prediction[0][index])
    
    return class_name, confidence_score

# --- UI LAYOUT ---
st.title("🚗 Numberplate Country Scanner")
st.write("Scan or upload a numberplate to determine its country of origin using our AI model. 🕵️")

# Sidebar
mode = st.sidebar.radio("Navigation", ["📷 Webcam Scanner", "📁 File Upload"])

col1, col2 = st.columns([2, 1])

with col1:
    if mode == "📷 Webcam Scanner":
        st.header("Webcam Scanner")
        st.write("Take a picture of the numberplate using your webcam.")
        
        if st.button("🔄 Reset Scanner"):
            reset_scan()
            st.rerun()
            
        camera_image = st.camera_input("Take a picture", key=f"camera_{st.session_state.scan_trigger}")
        
        if camera_image is not None:
            img_hash = hash(camera_image.getvalue())
            img = Image.open(camera_image).convert("RGB")
            
            with st.spinner("Analyzing numberplate..."):
                if st.session_state.last_processed_hash != img_hash:
                    country, conf = process_image(img)
                    if country:
                        add_to_history(img.copy(), country, conf)
                        st.session_state.last_processed_hash = img_hash
                    
            if len(st.session_state.history) > 0 and st.session_state.last_processed_hash == img_hash:
                latest = st.session_state.history[0]
                st.success(f"**Country Detected:** {latest['country']}")
                st.info(f"**Confidence Score:** {latest['confidence']*100:.2f}%")

    else:
        st.header("File Upload")
        st.write("Upload an image of the numberplate.")
        
        if st.button("🔄 Reset Upload"):
            reset_scan()
            st.rerun()
            
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state.scan_trigger}")
        
        if uploaded_file is not None:
            img_hash = hash(uploaded_file.getvalue())
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Analyzing uploaded image..."):
                if st.session_state.last_processed_hash != img_hash:
                    country, conf = process_image(img)
                    if country:
                        add_to_history(img.copy(), country, conf)
                        st.session_state.last_processed_hash = img_hash
                    
            if len(st.session_state.history) > 0 and st.session_state.last_processed_hash == img_hash:
                latest = st.session_state.history[0]
                st.success(f"**Country Detected:** {latest['country']}")
                st.info(f"**Confidence Score:** {latest['confidence']*100:.2f}%")

with col2:
    st.header("📜 Recent Scans")
    st.write("Last 5 scanned or uploaded plates.")
    
    if len(st.session_state.history) == 0:
        st.info("No scans yet. Start by capturing or uploading an image!")
    else:
        for idx, item in enumerate(st.session_state.history):
            with st.container():
                st.markdown('<div class="history-card">', unsafe_allow_html=True)
                st.image(item["image"], use_container_width=True)
                st.markdown(f"**{item['country']}** - {item['confidence']*100:.1f}%")
                st.caption(f"Scanned at {item['time']}")
                st.markdown('</div>', unsafe_allow_html=True)