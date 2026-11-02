import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import iid_client_subsets
from .model import create_pretrained_model, create_model_from_state
from .metrics import evaluate_state, calculate_metrics


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_state(state, path):
    torch.save(state, path)


def run_primary(train_dataset, val_loader, test_loader, output_dir, checkpoint_dir):
    """Reproduce notebook cells 3-4: separate initialization, central 3 epochs, FedAvg 2 rounds."""
    from torch.utils.data import DataLoader
    device = _device()
    batch_size = 8
    criterion = nn.CrossEntropyLoss()
    records = []

    # Centralized track: 3 epochs, best validation checkpoint.
    model = create_pretrained_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    start = time.time()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.view(-1).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        vy, vp = evaluate_state(model.state_dict(), val_loader, device)
        val_acc = (vy == vp).mean()
        print(f"Primary centralized epoch {epoch+1}/3: val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(model.state_dict())

    central_seconds = time.time() - start
    cy, cp = evaluate_state(best_state, test_loader, device)
    central_metrics = calculate_metrics(cy, cp)
    _save_state(best_state, f"{checkpoint_dir}/primary_centralized.pth")
    records.append({"model": "Centralized ViT", "time_seconds": central_seconds, **central_metrics})
    del model, optimizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Federated track: separate pretrained initialization, 3 IID clients, 2 rounds.
    torch.manual_seed(42)
    np.random.seed(42)
    clients = iid_client_subsets(train_dataset, 3, seed=42, use_torch_permutation=True)
    global_model = create_pretrained_model()
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    del global_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    fed_history = []
    start = time.time()
    for rnd in range(2):
        client_states, client_sizes = [], []
        for cid, dataset in enumerate(clients):
            loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
            local = create_model_from_state(global_state, device)
            optimizer = torch.optim.AdamW(local.parameters(), lr=1e-5, weight_decay=0.01)
            local.train()
            for x, y in loader:
                x, y = x.to(device), y.view(-1).long().to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = criterion(local(x), y)
                loss.backward()
                optimizer.step()
            client_states.append({k: v.detach().cpu().clone() for k, v in local.state_dict().items()})
            client_sizes.append(len(dataset))
            del local, optimizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        total = sum(client_sizes)
        global_state = {
            key: sum(client_states[i][key] * (client_sizes[i] / total) for i in range(3))
            if torch.is_floating_point(global_state[key]) else client_states[0][key].clone()
            for key in global_state
        }
        vy, vp = evaluate_state(global_state, val_loader, device)
        val_acc = (vy == vp).mean()
        fed_history.append({"round": rnd + 1, "validation_accuracy": float(val_acc)})
        print(f"Primary FedAvg round {rnd+1}/2: val_acc={val_acc:.4f}")

    fed_seconds = time.time() - start
    fy, fp = evaluate_state(global_state, test_loader, device)
    fed_metrics = calculate_metrics(fy, fp)
    _save_state(global_state, f"{checkpoint_dir}/primary_fedavg.pth")
    records.append({"model": "Federated FedAvg", "time_seconds": fed_seconds, **fed_metrics})

    pd.DataFrame(records).drop(columns=["confusion_matrix"]).to_csv(
        f"{output_dir}/primary_rerun_results.csv", index=False
    )
    print("Primary experiment:", records)
    print("FedAvg validation history:", fed_history)


def run_matched(train_dataset, val_loader, test_loader, output_dir, checkpoint_dir, seed=42):
    """Reproduce notebook cell 8: same initial weights and validation-based checkpoint selection."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = _device()
    base = create_pretrained_model()
    initial = copy.deepcopy(base.state_dict())
    del base
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    clients = iid_client_subsets(train_dataset, 3, seed=seed, use_torch_permutation=False)

    # Centralized, two epochs.
    start = time.time()
    central = create_model_from_state(initial, device)
    opt = torch.optim.AdamW(central.parameters(), lr=1e-5, weight_decay=0.01)
    best_c, best_c_val = copy.deepcopy(initial), -1
    for epoch in range(2):
        central.train()
        for x, y in train_loader:
            x, y = x.to(device), y.view(-1).long().to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(central(x), y)
            loss.backward()
            opt.step()
        state = copy.deepcopy(central.state_dict())
        vy, vp = evaluate_state(state, val_loader, device)
        val_acc = (vy == vp).mean()
        print(f"Matched centralized epoch {epoch+1}: val_acc={val_acc:.4f}")
        if val_acc > best_c_val: best_c_val, best_c = val_acc, state
    central_seconds = time.time() - start
    del central, opt
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # FedAvg, two rounds, one local epoch per client.
    start = time.time()
    global_state = copy.deepcopy(initial)
    best_f, best_f_val = copy.deepcopy(initial), -1
    for rnd in range(2):
        states, sizes = [], []
        for ds in clients:
            local = create_model_from_state(global_state, device)
            opt = torch.optim.AdamW(local.parameters(), lr=1e-5, weight_decay=0.01)
            local.train()
            for x, y in DataLoader(ds, batch_size=8, shuffle=True, num_workers=2):
                x, y = x.to(device), y.view(-1).long().to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(local(x), y)
                loss.backward()
                opt.step()
            states.append({k: v.detach().cpu().clone() for k, v in local.state_dict().items()})
            sizes.append(len(ds))
            del local, opt
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        total = sum(sizes)
        global_state = {
            k: sum(states[i][k] * (sizes[i] / total) for i in range(3))
            for k in global_state
        }
        vy, vp = evaluate_state(global_state, val_loader, device)
        val_acc = (vy == vp).mean()
        print(f"Matched FedAvg round {rnd+1}: val_acc={val_acc:.4f}")
        if val_acc > best_f_val: best_f_val, best_f = val_acc, copy.deepcopy(global_state)
    fed_seconds = time.time() - start

    results = []
    for name, state, seconds in [
        ("Centralized ViT", best_c, central_seconds),
        ("Federated FedAvg", best_f, fed_seconds),
    ]:
        y, p = evaluate_state(state, test_loader, device)
        metrics = calculate_metrics(y, p)
        _save_state(state, f"{checkpoint_dir}/matched_{'centralized' if name.startswith('Centralized') else 'fedavg'}.pth")
        results.append({"model": name, "time_seconds": seconds, **metrics})
    pd.DataFrame(results).drop(columns=["confusion_matrix"]).to_csv(
        f"{output_dir}/matched_rerun_results.csv", index=False
    )
    print("Matched comparison:", results)


def run_repeats(train_dataset, val_loader, test_loader, output_dir, checkpoint_dir, seeds=(7, 21)):
    """Reproduce notebook cell 11: two extra seeded matched experiments."""
    device = _device()
    criterion = nn.CrossEntropyLoss()
    all_rows = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        base = create_pretrained_model()
        initial = copy.deepcopy(base.state_dict())
        del base
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        def build(state):
            return create_model_from_state(state, device)

        def seeded_loader(ds, shuffle=False, seed_value=seed):
            gen = torch.Generator().manual_seed(seed_value)
            return DataLoader(ds, batch_size=8, shuffle=shuffle, num_workers=2, generator=gen)

        # Centralized two epochs.
        start = time.time()
        model = build(initial)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        best_c, best_val = copy.deepcopy(initial), -1
        for epoch in range(2):
            model.train()
            for x, y in seeded_loader(train_dataset, True, seed + epoch):
                x, y = x.to(device), y.view(-1).long().to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
            state = copy.deepcopy(model.state_dict())
            vy, vp = evaluate_state(state, val_loader, device)
            va = (vy == vp).mean()
            if va > best_val: best_val, best_c = va, state
        c_time = time.time() - start
        del model, opt
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Three IID clients, two FedAvg rounds.
        clients = iid_client_subsets(train_dataset, 3, seed=seed, use_torch_permutation=False)
        global_state = copy.deepcopy(initial)
        best_f, best_val = copy.deepcopy(initial), -1
        start = time.time()
        for rnd in range(2):
            states, sizes = [], []
            for cid, ds in enumerate(clients):
                local = build(global_state)
                opt = torch.optim.AdamW(local.parameters(), lr=1e-5, weight_decay=0.01)
                local.train()
                for x, y in seeded_loader(ds, True, seed + rnd * 10 + cid):
                    x, y = x.to(device), y.view(-1).long().to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = criterion(local(x), y)
                    loss.backward()
                    opt.step()
                states.append({k: v.detach().cpu().clone() for k, v in local.state_dict().items()})
                sizes.append(len(ds))
                del local, opt
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            total = sum(sizes)
            global_state = {k: sum(states[i][k] * (sizes[i] / total) for i in range(3)) for k in global_state}
            vy, vp = evaluate_state(global_state, val_loader, device)
            va = (vy == vp).mean()
            if va > best_val: best_val, best_f = va, copy.deepcopy(global_state)
        f_time = time.time() - start

        for name, state, seconds in [
            ("Centralized ViT", best_c, c_time),
            ("Federated FedAvg", best_f, f_time),
        ]:
            y, p = evaluate_state(state, test_loader, device)
            m = calculate_metrics(y, p)
            all_rows.append({
                "seed": seed, "model": name,
                "test_accuracy": m["test_accuracy"],
                "precision_class1": m["test_precision_class1"],
                "recall_class1": m["test_recall_class1"],
                "f1_class1": m["test_f1_class1"],
                "time_seconds": seconds,
            })
        print(f"Completed repeatability seed {seed}")

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{output_dir}/reproducibility_rerun_results.csv", index=False)
    print(df.to_string(index=False))
    print(df.groupby("model")[["test_accuracy", "precision_class1", "recall_class1", "f1_class1"]].agg(["mean", "std"]).round(4))


def run_all(experiment="all"):
    import argparse
    from pathlib import Path
    output_dir, checkpoint_dir = "results", "checkpoints"
    Path(output_dir).mkdir(exist_ok=True)
    Path(checkpoint_dir).mkdir(exist_ok=True)
    from .data import load_datasets, make_loaders
    train, val, test = load_datasets()
    _, val_loader, test_loader = make_loaders(train, val, test, batch_size=8)
    if experiment in ("all", "primary"):
        run_primary(train, val_loader, test_loader, output_dir, checkpoint_dir)
    if experiment in ("all", "matched"):
        run_matched(train, val_loader, test_loader, output_dir, checkpoint_dir)
    if experiment in ("all", "repeats"):
        run_repeats(train, val_loader, test_loader, output_dir, checkpoint_dir)
