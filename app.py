from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# The app.py file lives at the repository root, alongside src/ and checkpoints/.
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.model import create_model_from_state

CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "primary_fedavg.pth"
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="MedAI | Federated Diagnosis",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #0b1220; color: #f1f5f9; }
[data-testid="stSidebar"] { background-color: #111c30; }
h1, h2, h3 { color: #e2e8f0 !important; }
p, label { color: #cbd5e1; }
.metric-card {
    background: #17243a; border: 1px solid #293b55;
    border-radius: 14px; padding: 20px; margin-bottom: 12px;
}
.metric-value { font-size: 27px; font-weight: 700; color: #67e8f9; }
.metric-label { font-size: 13px; color: #a5b4c7; }
.hero {
    background: linear-gradient(120deg, #12304a, #17243a);
    padding: 28px; border-radius: 18px; border: 1px solid #31506c;
    margin-bottom: 25px;
}
.small-note { color: #94a3b8; font-size: 13px; }
div.stButton > button {
    background-color: #0891b2; color: white; border: none;
    border-radius: 9px; font-weight: 600;
}
div.stButton > button:hover { background-color: #0e7490; color: white; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {CHECKPOINT_PATH}. "
            "Add checkpoints/primary_fedavg.pth locally; the checkpoint "
            "is intentionally not stored in GitHub because it is large."
        )

    state = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model = create_model_from_state(state, DEVICE)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


st.sidebar.title("🩺 MedAI")
st.sidebar.caption("Federated Medical Image Research")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "X-ray Analysis", "Model Performance", "Methodology"],
)
st.sidebar.divider()
st.sidebar.markdown("### Model Information")
st.sidebar.write("Architecture: ViT-B/16")
st.sidebar.write("Learning: Federated Learning")
st.sidebar.write("Aggregation: FedAvg")
st.sidebar.write("Clients: 3")
st.sidebar.write("Communication rounds: 2")
st.sidebar.divider()
st.sidebar.caption("Academic research prototype | Not for clinical use")


if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>Privacy-Preserving Medical Image Diagnosis</h1>
        <p>An AI research prototype combining Vision Transformers and
        Federated Learning for pneumonia image classification.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Research Performance Overview")
    cols = st.columns(4)
    metrics = [
        ("91.19%", "Test Accuracy"),
        ("93.27%", "Pneumonia F1 Score"),
        ("97.69%", "Pneumonia Recall"),
        ("3", "Federated Clients"),
    ]
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}'
                f'</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)
    with left:
        st.subheader("Project Overview")
        st.write(
            "This project explores Vision Transformers with simulated "
            "Federated Learning for medical image classification. Three "
            "clients train locally, and their model updates are aggregated "
            "using Federated Averaging."
        )
    with right:
        st.subheader("Technology Stack")
        st.markdown("""
        - **Python:** Programming language
        - **PyTorch:** Deep learning framework
        - **ViT-B/16:** Image classification model
        - **FedAvg:** Federated aggregation
        - **PneumoniaMNIST:** Medical image dataset
        - **Streamlit:** Interactive web interface
        """)
    st.info(
        "Research demonstration only. Predictions must not be used "
        "for medical decisions."
    )


elif page == "X-ray Analysis":
    st.title("🩻 Chest X-ray Analysis")
    st.write("Upload an image to test the trained ViT model.")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Input chest X-ray", width=350)
        with col2:
            st.subheader("Image Information")
            st.write(f"Original dimensions: {image.width} × {image.height}")
            st.write("Input channels: Grayscale")
            st.write("Model input: 224 × 224")

        if st.button("🔍 Analyze X-ray", use_container_width=True):
            try:
                with st.spinner("Loading model and analyzing image..."):
                    model = load_model()
                    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0]

                    predicted_class = int(probabilities.argmax().item())
                    confidence = float(probabilities[predicted_class].item() * 100)

                st.divider()
                st.subheader("Analysis Result")
                prediction = "Normal" if predicted_class == 0 else "Pneumonia"
                if predicted_class == 0:
                    st.success("Prediction: NORMAL")
                else:
                    st.error("Prediction: PNEUMONIA")

                st.metric("Model confidence", f"{confidence:.2f}%")
                st.write("### Class probabilities")
                normal_prob = float(probabilities[0].item() * 100)
                pneumonia_prob = float(probabilities[1].item() * 100)
                st.write(f"Normal: {normal_prob:.2f}%")
                st.progress(normal_prob / 100)
                st.write(f"Pneumonia: {pneumonia_prob:.2f}%")
                st.progress(pneumonia_prob / 100)

                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, {
                    "Prediction": prediction,
                    "Confidence": f"{confidence:.2f}%",
                })
                st.caption(
                    "Experimental model output only. This is not a diagnosis."
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    if st.session_state.get("history"):
        st.divider()
        st.subheader("Recent Prediction History")
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
        )


elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.write("Results from the completed Colab test evaluation.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", "91.19%")
    col2.metric("Precision", "89.23%")
    col3.metric("Recall", "97.69%")
    col1, col2 = st.columns(2)
    col1.metric("F1 Score", "93.27%")
    col2.metric("Round 2 Validation Accuracy", "96.37%")

    st.divider()
    st.subheader("Confusion Matrix")
    confusion = pd.DataFrame(
        [[188, 46], [9, 381]],
        index=["Actual Normal", "Actual Pneumonia"],
        columns=["Predicted Normal", "Predicted Pneumonia"],
    )
    st.dataframe(confusion, use_container_width=True)
    st.caption(
        "Rows represent actual classes and columns represent predicted classes. "
        "Test set: 624 images."
    )
    st.info(
        "These metrics describe the recorded Colab experiment. "
        "Do not mix them with the separate notebook-recorded experiment tracks."
    )


elif page == "Methodology":
    st.title("🧠 Research Methodology")
    st.subheader("1. Dataset")
    st.write(
        "PneumoniaMNIST was used for binary classification of Normal "
        "and Pneumonia chest X-ray images."
    )
    st.subheader("2. Image Preprocessing")
    st.markdown("""
    - Resize images to 224 × 224 pixels.
    - Convert grayscale images into three channels.
    - Apply ImageNet normalization.
    """)
    st.subheader("3. Vision Transformer")
    st.write(
        "An ImageNet-pretrained ViT-B/16 model was used. Its final "
        "classification head was modified for two classes: Normal and Pneumonia."
    )
    st.subheader("4. Federated Learning")
    st.markdown("""
    - Divide training data into three IID clients.
    - Each client performs one local training epoch.
    - Send locally updated model weights to the server.
    - Aggregate weights using Federated Averaging (FedAvg).
    - Repeat for two communication rounds.
    """)
    st.subheader("5. Evaluation")
    st.write(
        "The model was evaluated using accuracy, precision, recall, "
        "F1 score and a confusion matrix."
    )
    st.warning(
        "Federated learning was simulated on a single machine. FedAvg alone "
        "does not guarantee differential privacy or secure aggregation."
    )


st.divider()
st.markdown(
    '<p class="small-note" style="text-align:center;">MedAI | '
    'Privacy-Preserving Medical Image Diagnosis<br>Academic Research '
    'Prototype | Vision Transformer + Federated Learning</p>',
    unsafe_allow_html=True,
)
