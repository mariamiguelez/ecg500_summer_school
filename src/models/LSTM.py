import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LSTMClassifierAdapter:
    # Used to adapt the format of the repo to the LSTM architecture
    def __init__(self, model, classes: np.ndarray, device, scaler: StandardScaler | None = None):
        self.model = model
        self.classes = classes
        self.device = device
        self.scaler = scaler

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            logits = self.model(x_tensor)
            class_idx = torch.argmax(logits, dim=1).cpu().numpy()
        return self.classes[class_idx]


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        # maps that summary (last output) to one score per class (logits)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)


def fit_lstm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    use_scaling: bool = False,
) -> LSTMClassifierAdapter:

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Save classes
    classes = np.sort(np.unique(y_train))
    class_to_idx = {label: i for i, label in enumerate(classes)}

    # Save the index liked to the label
    y_idx = np.array([class_to_idx[label] for label in y_train], dtype=np.int64)

    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, 1)).reshape(x_train.shape)

    # Turns (time, samples) into (time, samples, 1), where: input_size = 1 (one scalar amplitude per time step)
    x_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)

    # Labels are converted to class indices -> CrossEntropyLoss requires integer class ids (not hot one encoded)
    y_tensor = torch.tensor(y_idx, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define model optimizer and criterion
    model = LSTMClassifier(
        hidden_size=64,  # TODO: add to config
        num_layers=1,
        num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    weight_tensor = torch.tensor([0.584,0.354,0.02,0.038,0.004])  # TODO: add to config

    criterion = nn.CrossEntropyLoss(weight= 1 - weight_tensor)

    model.train()

    train_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        print(f"Epoch: {epoch + 1:3d} | " f"Train loss: {train_loss:.8f}")

    # Plot loss curve
    plt.plot(train_losses)
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

    return LSTMClassifierAdapter(model=model, classes=classes, device=device, scaler=scaler)
