import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from .model import create_model_from_state


def evaluate_state(state, loader, device):
    model = create_model_from_state(state, device)
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            logits = model(images)
            predictions.extend(logits.argmax(1).cpu().tolist())
            truth.extend(labels.cpu().tolist())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(truth), np.asarray(predictions)


def calculate_metrics(y_true, y_pred):
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision_class1": precision_score(y_true, y_pred, zero_division=0),
        "test_recall_class1": recall_score(y_true, y_pred, zero_division=0),
        "test_f1_class1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
