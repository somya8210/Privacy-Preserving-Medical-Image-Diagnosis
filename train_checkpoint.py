"""Train the repository's primary Federated ViT experiment and save an inference checkpoint."""
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import load_datasets, make_loaders, iid_client_subsets
from src.model import create_pretrained_model, create_model_from_state
from src.metrics import evaluate_state, calculate_metrics

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cpu":
        print("WARNING: ViT-B/16 training can be slow on CPU. A CUDA GPU/Colab runtime is recommended.")

    train, val, test = load_datasets()
    _, val_loader, test_loader = make_loaders(train, val, test, batch_size=8)
    clients = iid_client_subsets(train, num_clients=3, seed=42, use_torch_permutation=True)
    criterion = nn.CrossEntropyLoss()

    base = create_pretrained_model()
    global_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    del base

    for rnd in range(2):
        states, sizes = [], []
        for client_id, dataset in enumerate(clients, start=1):
            print(f"Round {rnd+1}/2 — client {client_id}/3 — {len(dataset)} images")
            local = create_model_from_state(global_state, device)
            optimizer = torch.optim.AdamW(local.parameters(), lr=1e-5, weight_decay=0.01)
            loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
            local.train()
            for images, labels in loader:
                images = images.to(device)
                labels = labels.view(-1).long().to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(local(images), labels)
                loss.backward()
                optimizer.step()
            states.append({k: v.detach().cpu().clone() for k, v in local.state_dict().items()})
            sizes.append(len(dataset))
            del local, optimizer

        total = sum(sizes)
        global_state = {
            key: sum(states[i][key] * (sizes[i] / total) for i in range(len(states)))
            if torch.is_floating_point(global_state[key]) else states[0][key].clone()
            for key in global_state
        }
        y_true, y_pred = evaluate_state(global_state, val_loader, device)
        print(f"Round {rnd+1} validation accuracy: {(y_true == y_pred).mean():.4f}")

    y_true, y_pred = evaluate_state(global_state, test_loader, device)
    print("Final test metrics:", calculate_metrics(y_true, y_pred))

    out = Path("checkpoints")
    out.mkdir(exist_ok=True)
    target = out / "primary_fedavg.pth"
    torch.save(global_state, target)
    print(f"Saved inference checkpoint: {target.resolve()}")

if __name__ == "__main__":
    main()
