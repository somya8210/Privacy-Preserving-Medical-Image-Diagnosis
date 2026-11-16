# Working demo: Streamlit X-ray inference

This adds a browser-based research demo around the existing ViT/FedAvg experiment.

## What it does

- Upload a PNG/JPG chest X-ray.
- Runs inference using the repository's two-class ViT-B/16 model.
- Shows the model's Normal/Pneumonia scores.
- Displays the repository's recorded centralized-vs-federated experiment tables.
- Explains the three-client, two-round FedAvg simulation.

## Important: generate the trained checkpoint first

The repository currently tracks notebook experiment results but does not include a trained `.pth` checkpoint. The app deliberately does not fabricate an inference result when the checkpoint is missing.

Use Python 3.10 or 3.11 and preferably a CUDA-enabled Google Colab/runtime:

```bash
pip install -r requirements-app.txt
python train_checkpoint.py
```

Training downloads PneumoniaMNIST through MedMNIST and uses ImageNet-pretrained torchvision ViT-B/16 weights. It runs the two-round, three-client FedAvg training procedure and saves:

```
checkpoints/primary_fedavg.pth
```

Then start the UI:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## CPU note

ViT-B/16 is computationally demanding. CPU training may be very slow; a GPU runtime is recommended. The checkpoint must be generated in the environment where the app will run, or copied into its `checkpoints/` folder.

## Medical and privacy limitations

This is an academic research prototype, not a clinical diagnostic tool. Do not use it to make medical decisions. Do not upload identifiable patient information.

The FL implementation simulates three clients sequentially on one machine. Raw training images remain in the local dataset partitions during simulation, but model updates are exchanged. FedAvg by itself does not guarantee differential privacy, secure aggregation, or anonymity. The demo's single-image inference also does not execute a live multi-hospital federated training round.
