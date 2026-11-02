# Privacy-Preserving Medical Image Diagnosis using Federated Learning and Vision Transformers

A research implementation exploring medical image classification with a Vision Transformer (ViT) and simulated Federated Learning (FL) using Federated Averaging (FedAvg).

## Project overview

This project compares a centrally trained ViT model with a federated setup in which the training data is divided among three IID clients. Each client performs local training, sends model parameters to the server, and the server aggregates the updates using FedAvg. The notebook records two communication rounds.

- **Dataset:** PneumoniaMNIST
- **Model:** ImageNet-pretrained ViT-B/16
- **Framework:** PyTorch
- **Federated setup:** 3 simulated IID clients, FedAvg
- **Optimizer:** AdamW

> **Scope note:** This repository documents a single-machine federated-learning simulation. It does not demonstrate deployment across independent hospitals or prove formal privacy guarantees such as differential privacy or secure aggregation. Model updates are exchanged in the simulation; privacy claims should therefore be appropriately limited.

## Repository structure

```text
.
├── notebooks/
│   └── Federated_ViT_Medical_Diagnosis.ipynb
├── src/
│   ├── data.py
│   ├── experiments.py
│   ├── metrics.py
│   └── model.py
├── results/
│   ├── README.md
│   ├── notebook_primary_results.csv
│   ├── notebook_matched_comparison.csv
│   └── notebook_reproducibility_runs.csv
├── docs/
│   └── methodology.md
├── config.json
├── requirements.txt
├── run_experiments.py
└── .gitignore
```

## Recorded results

The uploaded notebook contains separate experiment tracks. Keep them separate when reporting results:

- Primary run: centralized test accuracy **86.54%**; federated test accuracy **91.67%**.
- Matched-initialization comparison: centralized **89.42%**; federated **86.38%**.

These are notebook-recorded outputs, not results independently re-executed during repository organization. See the CSV files in `results/` and the notebook for context.

## Getting started

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
4. Open `notebooks/Federated_ViT_Medical_Diagnosis.ipynb` in Google Colab or Jupyter.

The notebook is the primary record of the experiment. The scripts in `src/` and `run_experiments.py` provide organized project code; check configuration and dataset paths before running locally.

## Data and reproducibility

PneumoniaMNIST is a public dataset distributed through MedMNIST. Follow its official terms and instructions when obtaining the data. Do not commit private patient data, credentials, model checkpoints, or large raw datasets to this repository.

## Limitations

- Federated clients are simulated on one machine and use IID data partitioning.
- The experiment uses a small number of clients and communication rounds.
- Results should not be interpreted as clinical validation or evidence of deployment-ready medical diagnosis.
- A FedAvg simulation alone does not provide formal privacy guarantees.

## Citation

Add the final paper citation here once its publication details are confirmed.
