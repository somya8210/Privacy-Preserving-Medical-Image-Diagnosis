import json
from pathlib import Path

import streamlit as st
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.model import create_model_from_state

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "checkpoints" / "primary_fedavg.pth"
RESULTS = ROOT / "results"

st.set_page_config(page_title="Privacy-Preserving Medical Image Diagnosis", page_icon="🩺", layout="wide")

st.title("🩺 Privacy-Preserving Medical Image Diagnosis")
st.caption("Vision Transformer (ViT-B/16) • Federated Averaging • PneumoniaMNIST")

st.warning(
    "Research demonstration only. This tool is not a medical device and cannot diagnose, "
    "exclude, or treat pneumonia. Never use its output for clinical decisions."
)

@st.cache_resource
def load_model():
    if not CHECKPOINT.exists():
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = create_model_from_state(state, device)
    model.eval()
    return model, device

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

tab1, tab2, tab3 = st.tabs(["Image analysis", "Experiment comparison", "How federated learning works"])

with tab1:
    st.subheader("Upload a chest X-ray")
    uploaded = st.file_uploader("Choose a PNG or JPG image", type=["png", "jpg", "jpeg"])
    model, device = load_model()

    if model is None:
        st.info(
            "The trained checkpoint is not present yet. Run the training command in the README "
            "to generate checkpoints/primary_fedavg.pth, then restart the app."
        )
    elif uploaded is not None:
        image = Image.open(uploaded).convert("L")
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
        with st.spinner("Running ViT inference..."):
            x = preprocess(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(x)
                probabilities = torch.softmax(logits, dim=1)[0].cpu()
        labels = ["Normal", "Pneumonia"]
        pred = int(torch.argmax(probabilities).item())
        with right:
            st.metric("Model output", labels[pred])
            st.write(f"Normal score: **{probabilities[0].item()*100:.2f}%**")
            st.write(f"Pneumonia score: **{probabilities[1].item()*100:.2f}%**")
            st.progress(float(probabilities[pred].item()))
            st.caption("Scores are model probabilities, not calibrated clinical risk estimates.")
    elif uploaded is None:
        st.write("Upload a de-identified chest X-ray to run the trained research model.")

with tab2:
    st.subheader("Notebook-recorded experiment results")
    st.caption("These are results recorded in the repository notebook, not freshly evaluated by this app.")
    primary_path = RESULTS / "notebook_primary_results.csv"
    matched_path = RESULTS / "notebook_matched_comparison.csv"
    if primary_path.exists():
        st.markdown("**Primary experiment (separate model initializations)**")
        df = pd.read_csv(primary_path)
        st.dataframe(df, use_container_width=True, hide_index=True)
    if matched_path.exists():
        st.markdown("**Matched-initialization comparison**")
        df2 = pd.read_csv(matched_path)
        st.dataframe(df2, use_container_width=True, hide_index=True)
    if not primary_path.exists() and not matched_path.exists():
        st.info("No recorded comparison CSVs found in results/.")

with tab3:
    st.subheader("Three simulated clients and one coordinating server")
    st.markdown("""
    1. The PneumoniaMNIST training split is divided into three approximately equal IID subsets.
    2. Each client receives the current global ViT weights and trains locally for one epoch.
    3. Clients send model parameters back to the coordinating server; raw training images are not exchanged in this simulation.
    4. The server performs sample-count-weighted Federated Averaging (FedAvg).
    5. The process repeats for two communication rounds.
    """)
    st.info(
        "This repository simulates clients in one process on one machine. FedAvg alone does not "
        "provide differential privacy or secure aggregation, and this demo does not establish "
        "deployment across independent hospitals."
    )

st.divider()
st.caption("Model: ImageNet-pretrained torchvision ViT-B/16 • 224×224 input • AdamW • 2 output classes")
