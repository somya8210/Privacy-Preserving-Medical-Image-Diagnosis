# Methodology implemented in the uploaded notebook

## Dataset and preprocessing
The notebook loads the PneumoniaMNIST train, validation and test splits through MedMNIST. The notebook output records 4,708 training images, 524 validation images and 624 test images. Images are originally 28×28, resized to 224×224, converted to three-channel grayscale tensors and normalized with ImageNet mean and standard deviation.

## Vision Transformer
The implementation uses `torchvision.models.vit_b_16` with ImageNet pretrained weights. The original classification head is replaced with a linear layer containing two output units. The loss is cross entropy.

## Centralized baseline
The primary notebook experiment uses AdamW with learning rate 1e-5 and weight decay 0.01, batch size 8, and three epochs. The checkpoint with the highest validation accuracy is evaluated on the held-out test set.

## Federated simulation
The training split is partitioned into three approximately equal IID subsets (1,570, 1,569 and 1,569 examples in the primary run). Each client receives the current global model, performs one local epoch using AdamW, and returns its model state. The server aggregates floating-point parameters by sample-count-weighted FedAvg. This is repeated for two communication rounds. Validation accuracy is recorded after each round, and the final global model is evaluated on the test split.

## Additional matched comparison
The notebook contains a second experiment in which both methods begin from the same pretrained model state. The centralized model trains for two epochs. FedAvg uses three IID clients, two rounds and one local epoch per client. Validation accuracy selects the checkpoint; test metrics are then computed for the selected model.

## Additional repeatability runs
The notebook repeats the matched setup for seeds 7 and 21. These are two additional runs (n=2), not a large-scale statistical study.

## Metrics
Accuracy is overall correct predictions divided by all test examples. Precision, recall and F1 are computed for class label 1 using scikit-learn defaults for binary classification. Confusion matrices use the class order 0, 1.

## Limitations
- Clients are simulated sequentially in a single process and share the same runtime/hardware.
- The IID partition does not model realistic hospital-level non-IID distributions.
- FedAvg does not itself provide differential privacy or secure aggregation.
- Only PneumoniaMNIST binary classification is implemented in this notebook.
- Two additional seeds are insufficient for strong statistical claims.
- Results and wall-clock times can vary across hardware and software environments.
