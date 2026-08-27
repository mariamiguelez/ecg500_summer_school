from __future__ import annotations

import copy

import numpy as np
import pywt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class WaveletCNNClassifier(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(start_dim=1)
        return self.classifier(features)


class WaveletCNNAdapter:
    def __init__(
        self,
        model: WaveletCNNClassifier,
        classes: np.ndarray,
        scales: np.ndarray,
        wavelet: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.classes = classes
        self.scales = scales
        self.wavelet = wavelet
        self.device = device
        self.scaler = None

    def _transform_batch(self, x: np.ndarray) -> np.ndarray:
        transformed = []
        for series in x:
            coefficients, _ = pywt.cwt(series, self.scales, self.wavelet)
            transformed.append(
                np.stack([coefficients.real, coefficients.imag], axis=0).astype(np.float32)
            )
        return np.stack(transformed, axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        wavelet_data = self._transform_batch(x)
        x_tensor = torch.tensor(wavelet_data, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            class_indices = torch.argmax(self.model(x_tensor), dim=1).cpu().numpy()
        return self.classes[class_indices]


def fit_cnn_wavelet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    random_state: int = 42,
    loss_weights: list[float] | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    wavelet: str = "morl",
    max_scale: int = 64,
) -> WaveletCNNAdapter:
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    classes = np.sort(np.unique(y_train))
    class_to_index = {label: index for index, label in enumerate(classes)}
    y_train_indices = np.array([class_to_index[label] for label in y_train])

    if x_val is None or y_val is None:
        x_train, x_val, y_train_indices, y_val_indices = train_test_split(
            x_train,
            y_train_indices,
            stratify=y_train_indices,
            test_size=0.1,
            random_state=random_state,
        )
    else:
        y_val_indices = np.array([class_to_index[label] for label in y_val])

    scales = np.arange(1, max_scale + 1)

    def transform_batch(x: np.ndarray) -> np.ndarray:
        transformed = []
        for series in x:
            coefficients, _ = pywt.cwt(series, scales, wavelet)
            transformed.append(
                np.stack([coefficients.real, coefficients.imag], axis=0).astype(np.float32)
            )
        return np.stack(transformed, axis=0)

    train_data = transform_batch(x_train)
    val_data = transform_batch(x_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveletCNNClassifier(n_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = torch.tensor(
        loss_weights if loss_weights is not None else np.ones(len(classes)),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(y_train_indices, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_data, dtype=torch.float32),
            torch.tensor(y_val_indices, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(x_batch), y_batch).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Epoch: {epoch + 1:3d} | Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f}"
        )

    model.load_state_dict(best_state)
    return WaveletCNNAdapter(
        model=model,
        classes=classes,
        scales=scales,
        wavelet=wavelet,
        device=device,
    )
