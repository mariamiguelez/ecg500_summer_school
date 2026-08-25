import torch
import time
import copy
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    loss_weights: np.ndarray = None,
    random_state: int = 42,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    use_scaling: bool = False,
    hidden_size: int = 64,
    n_layers: int = 1

) -> LSTMClassifierAdapter:

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # If there is no validation set provided take a 0.1 portion from the train set
    if x_val is None or y_val is None:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            stratify=y_train,
            test_size=0.1,
            random_state=random_state,
        )

    # Save classes
    classes = np.sort(np.unique(np.concatenate([y_train, y_val])))
    class_to_idx = {label: i for i, label in enumerate(classes)}

    # Save the index liked to the label
    y_idx = np.array([class_to_idx[label] for label in y_train], dtype=np.int64)
    y_val_idx = np.array([class_to_idx[label] for label in y_val], dtype=np.int64)

    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, 1)).reshape(x_train.shape)
        x_val = scaler.transform(x_val.reshape(-1, 1)).reshape(x_val.shape)

    # Turns (time, samples) into (time, samples, 1), where: input_size = 1 (one scalar amplitude per time step)
    x_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)

    # Labels are converted to class indices -> CrossEntropyLoss requires integer class ids (not hot one encoded)
    y_tensor = torch.tensor(y_idx, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1)
    y_val_tensor = torch.tensor(y_val_idx, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define model optimizer and criterion
    model = LSTMClassifier(
        hidden_size=hidden_size,
        num_layers=n_layers,
        num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    weight_tensor = torch.tensor(loss_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight= weight_tensor)

    train_losses = []
    val_losses = []
    time_per_epoch = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_epoch = 0

    for epoch in range(epochs):
        start_time = time.time()  # Start time per epoch
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
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.8f} | "
            f"Val loss: {val_loss:.8f} | "
            f"Best val epoch: {best_val_epoch + 1}"
        )
        end_time = time.time()
        time_per_epoch.append(end_time - start_time)  # Appends the time per epoch
    average_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
    print(f"Average time per epoch: {average_time_per_epoch:.4f}s")

    # Plot loss curve
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()

    model.load_state_dict(best_state_dict)
    return LSTMClassifierAdapter(model=model, classes=classes, device=device, scaler=scaler)
