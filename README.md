# Privacy-Preserving Medical Image Diagnosis using Federated Learning and Vision Transformers

A research implementation exploring medical image classification with a Vision Transformer (ViT) and simulated Federated Learning (FL) using Federated Averaging (FedAvg).

## Project overview

This project compares a centrally trained ViT model with a federated setup in which the training data is divided among three IID clients. Each client performs local training, sends model parameters to the server, and the server aggregates the updates using FedAvg. The notebook records two communication rounds.

- **Dataset:** PneumoniaMNIST
- **Model:** ImageNet-pretrained ViT-B/16
- **Framework:** PyTorch
- **Federated setup:** 3 simulated IID clients, FedAvg
- **Optimizer:** AdamW
- **Demo:** Streamlit application in `app.py`

> **Scope note:** This repository documents a single-machine federated-learning simulation. It does not demonstrate deployment across independent hospitals or prove formal privacy guarantees such as differential privacy or secure aggregation. Model updates are exchanged in the simulation; privacy claims should therefore be appropriately limited.

## Repository structure

```text
.
├── app.py
├── notebooks/
│   └── Federated_ViT_Medical_Diagnosis.ipynb
├── src/
│   ├── data.py
│   ├── experiments.py
│   ├── metrics.py
│   └── model.py
├── results/
├── docs/
├── config.json
├── requirements.txt
├── run_experiments.py
└── .gitignore
```

## Run the Streamlit demo

1. Clone this repository:
   ```bash
   git clone https://github.com/somya8210/Privacy-Preserving-Medical-Image-Diagnosis.git
   cd Privacy-Preserving-Medical-Image-Diagnosis
   ```
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Obtain the trained checkpoint from your project owner/experiment storage and place it at:
   ```text
   checkpoints/primary_fedavg.pth
   ```
   The checkpoint is intentionally excluded from Git because it is approximately 327 MB. The app will show an explanatory error if the file is missing.
5. Launch the app:
   ```bash
   streamlit run app.py
   ```

The demo runs inference on CPU by default. It accepts JPG, JPEG, and PNG images. The app uses the same 224 × 224 resize, grayscale-to-three-channel conversion, and ImageNet normalization used by the model pipeline.

## Recorded results

The uploaded notebook contains separate experiment tracks. Keep them separate when reporting results:

- Primary run: centralized test accuracy **86.54%**; federated test accuracy **91.67%**.
- Matched-initialization comparison: centralized **89.42%**; federated **86.38%**.

A separate Colab demo evaluation recorded test accuracy **91.19%**, pneumonia precision **89.23%**, recall **97.69%**, and F1-score **93.27%** on 624 test images. These results belong to that specific run and should not be conflated with the notebook-recorded experiment tracks. See the CSV files in `results/` and the notebook for their context.

## Data and reproducibility

PneumoniaMNIST is a public dataset distributed through MedMNIST. Follow its official terms and instructions when obtaining the data. Do not commit private patient data, credentials, model checkpoints, or large raw datasets to this repository.

## Limitations

- Federated clients are simulated on one machine and use IID data partitioning.
- The experiment uses a small number of clients and communication rounds.
- Results should not be interpreted as clinical validation or evidence of deployment-ready medical diagnosis.
- A FedAvg simulation alone does not provide formal privacy guarantees.
- The Streamlit app is an academic research demonstration, not a medical diagnostic device.

## Citation

Add the final paper citation here once its publication details are confirmed.
