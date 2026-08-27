import os
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader

from mantis.architecture import MantisV2


# Anchored to the script directory so the checkpoint location does not
# depend on the process's current working directory.
_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_model.pth",
)
class MantisAdapter:
    # Used to adapt this model to the training pipeline interface
    def __init__(self, model, classes: np.ndarray, device, scaler: StandardScaler | None = None):
        self.model = model
        self.classes = classes
        self.device = device
        self.scaler = scaler

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            logits = self.model(x_tensor) #obtain logits from the linear prediction which can take any value in R
            # get the indice with the highest logit value as after transforming to back to
            # probabilities the same value will have the highest prob
            class_idx = torch.argmax(logits, dim=1).cpu().numpy()

        return self.classes[class_idx] # return the class taking the indice with the highes logit


def save_checkpoint_safely(
    state_dict: dict,
    path: str,
    max_retries: int = 5,
    retry_delay: float = 1.0,
) -> None:
    """Save a model checkpoint, tolerating transient Windows file locks.

    Saves to a temporary file first and atomically replaces the target,
    so a crash or lock never leaves a corrupted checkpoint behind.
    A file that is briefly locked by a virus scanner, sync client (e.g.
    OneDrive), or another process is retried instead of failing training.
    """

    tmp_path = f"{path}.tmp-{os.getpid()}"

    for attempt in range(max_retries):

        try:
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, path)
            return

        except (RuntimeError, OSError, PermissionError) as error:

            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if attempt == max_retries - 1:
                print(
                    f"Warning: could not save checkpoint to {path} "
                    f"after {max_retries} attempts ({error}). "
                    "Continuing training without saving this checkpoint."
                )
                return

            print(
                f"Warning: checkpoint save attempt {attempt + 1} failed "
                f"({error}). Retrying in {retry_delay:.1f}s..."
            )

            time.sleep(retry_delay)



# MantisV2 Models

class MantisV2FrozenClassifier(nn.Module):
    """MantisV2 classifier with frozen pretrained encoder."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        device: torch.device,
        target_length: int = 512,
    ):
        super().__init__()

        self.target_length = target_length

        # Load pretrained MantisV2 encoder.
        #
        # return_transf_layer=-1:
        #     Use the output of the final Transformer layer.
        #
        # output_token="cls_token":
        #     Use the final CLS token as sequence representation.
        self.encoder = MantisV2(
            device=str(device),
            return_transf_layer=-1,
            output_token="cls_token",
        )

        self.encoder = self.encoder.from_pretrained(
            "paris-noah/MantisV2"
        )

        # Freeze all pretrained encoder parameters.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # MantisV2 CLS embedding dimension = 256.
        embedding_dim = self.encoder.hidden_dim

        # Each sensor channel is encoded independently.
        #
        # RacketSports:
        # 6 channels * 256 embeddings = 1536 features.
        self.output_projection = nn.Linear(
            input_size * embedding_dim,
            n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input:
        # (B, T, C)
        #
        # RacketSports:
        # (B, 30, 6)

        # Change to channel-first representation
        print(x.shape)
        x = x.transpose(1, 2)   # (B, 30, 6) -> (B, 6, 30)

        # Resize signal length. Recommendation by the authors, because the mode model was trained on a sequence length of 512
        x = F.interpolate(
            x,
            size=self.target_length,
            mode="linear",
            align_corners=False,
        ) # (B, 30, 6) -> (B, 6, 512)

        # Move channels into batch dimension
        batch_size, n_channels, seq_len = x.shape

        x = x.reshape(
            batch_size * n_channels,
            1,
            seq_len,
        ) # (B, 6, 512) -> (B*6, 1, 512)


        # Frozen MantisV2 encoder
        # model.train() from the outer training loop would otherwise put the encoder into training mode.
        self.encoder.eval()

        with torch.no_grad():
            embeddings = self.encoder(x) # (B*6, 256)

        # Restore channel dimension
        embeddings = embeddings.reshape(
            batch_size,
            n_channels,
            -1,
        ) # (B*6, 256) -> (B, 6, 256)

        # Concatenate channel embeddings
        embeddings = embeddings.flatten(start_dim=1) # (B, 6, 256) -> (B, 1536)

        # Classification
        output = self.output_projection(embeddings) # (B, 1536) -> (B, 4)

        return output


class MantisV2FineTunedClassifier(nn.Module):
    """MantisV2 classifier with fine-tuned pretrained encoder."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        device: torch.device,
        target_length: int = 512,
    ):
        super().__init__()

        self.target_length = target_length

        # Load exactly the same pretrained MantisV2 encoder
        # representation as for the frozen model.
        self.encoder = MantisV2(
            device=str(device),
            return_transf_layer=-1,
            output_token="cls_token",
        )

        self.encoder = self.encoder.from_pretrained(
            "paris-noah/MantisV2"
        )

        # No parameters are frozen.
        # Gradients therefore propagate through MantisV2.

        embedding_dim = self.encoder.hidden_dim

        # RacketSports:
        # 6 channels * 256 embeddings = 1536 features.
        self.output_projection = nn.Linear(
            input_size * embedding_dim,
            n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input:
        # (B, T, C)
        #
        # RacketSports:
        # (B, 30, 6)

        # Change to channel-first representation
        x = x.transpose(1, 2)   # (B, 30, 6) -> (B, 6, 30)

        # Resize signal length. Recommendation by the authors, because the mode model was trained on a sequence length of 512
        x = F.interpolate(
            x,
            size=self.target_length,
            mode="linear",
            align_corners=False,
        ) # (B, 30, 6) -> (B, 6, 512)

        # Move channels into batch dimension
        batch_size, n_channels, seq_len = x.shape

        x = x.reshape(
            batch_size * n_channels,
            1,
            seq_len,
        ) # (B, 6, 512) -> (B*6, 1, 512)

        embeddings = self.encoder(x) # (B*6, 256)

        # Restore channel dimension
        embeddings = embeddings.reshape(
            batch_size,
            n_channels,
            -1,
        ) # (B*6, 256) -> (B, 6, 256)

        # Concatenate channel embeddings
        embeddings = embeddings.flatten(start_dim=1) # (B, 6, 256) -> (B, 1536)

        # Classification
        output = self.output_projection(embeddings) # (B, 1536) -> (B, 4)

        return output

class ECGDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def fit_mantisV2_classifier(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        random_state: int = 42,
        epochs: int = 200,
        input_size: int = 1,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        use_scaling: bool = False,
        loss_weights: np.ndarray | None = None,
        model_variant: str = "frozen",
        target_length: int = 512,
        plot_losses: bool = True,
    ) -> MantisV2FineTunedClassifier:

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    # If there is no validation set use 0.1 the split from the train set
    if x_val is None or y_val is None:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            stratify=y_train,
            test_size=0.1,
            random_state=random_state,
        )

    train_data, train_labels = x_train, y_train
    # Get unique classes
    classes = np.sort(np.unique(np.concatenate([y_train, y_val])))
    class_to_idx = {label: i for i, label in enumerate(classes)}
    train_labels_idx = np.array([class_to_idx[label] for label in train_labels], dtype=np.int64)
    val_data = x_val
    val_labels_idx = np.array([class_to_idx[label] for label in y_val], dtype=np.int64)

    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        train_data = scaler.fit_transform(
                train_data.reshape(-1, train_data.shape[-1])
            ).reshape(train_data.shape)
        val_data = scaler.transform(
                val_data.reshape(-1, val_data.shape[-1])
            ).reshape(val_data.shape)

    ### Initialize the dataset
    train_dataset = ECGDataset(data=train_data, labels=train_labels_idx)
    val_dataset = ECGDataset(data=val_data, labels=val_labels_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    ### Initialize model and optimizer
    if train_data.ndim == 2:
        input_size = 1
    else:
        input_size = train_data.shape[-1]
    n_classes = len(classes)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_variant_normalized = model_variant.strip().lower()
    if model_variant_normalized == "frozen":
        model_class = MantisV2FrozenClassifier
    elif model_variant_normalized in {"finetuned", "fine_tuned", "fine-tuned"}:
        model_class = MantisV2FineTunedClassifier
    else:
        raise ValueError(
            "model_variant must be either 'frozen' or 'finetuned'. "
            f"Got: {model_variant}"
        )

    model = model_class(
        input_size=input_size,
        n_classes=n_classes,
        device=device,
        target_length=target_length,
    ).to(device)

    model_name = model.__class__.__name__

    print(
        f"\nModel: {model_name}"
    )

    # Parameter count
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    optimizer = None # To omit the warning
    if model_name == "MantisV2FrozenClassifier":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), # To train only the parameters which require gradients
            lr=learning_rate,
        )

    elif model_name == "MantisV2FineTunedClassifier":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )


    if loss_weights is None:
        criterion = nn.CrossEntropyLoss()
    else:
        if len(loss_weights) != n_classes:
            raise ValueError(
                "loss_weights length must match number of classes. "
                f"Got {len(loss_weights)} weights for {n_classes} classes."
            )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(loss_weights, dtype=torch.float32, device=device)
        )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_val_epoch = 0

    time_per_epoch = []

    for epoch in range(epochs):

        start_time = time.time()

        model.train()

        train_loss = 0.0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            if x_batch.dim() == 2:
                x_batch = x_batch.unsqueeze(-1)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(x_batch)

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        train_losses.append(
            train_loss
        )

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for x_batch, y_batch in val_loader:

                x_batch = x_batch.to(device)
                if x_batch.dim() == 2:
                    x_batch = x_batch.unsqueeze(-1)
                y_batch = y_batch.to(device)

                predictions = model(
                    x_batch
                )

                loss = criterion(
                    predictions,
                    y_batch,
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        val_losses.append(
            val_loss
        )

        end_time = time.time()

        time_per_epoch.append(
            end_time - start_time
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_val_epoch = epoch

            save_checkpoint_safely(
                model.state_dict(),
                _CHECKPOINT_PATH,
            )

        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.8f} | "
            f"Val loss: {val_loss:.8f} | "
            f"Best Val loss: {best_val_loss:.8f} | "
            f"Best Val epoch: {best_val_epoch + 1}"
        )

    if plot_losses:
        plt.figure()
        plt.plot(
            train_losses,
            label="Train Loss",
        )

        plt.plot(
            val_losses,
            label="Validation Loss",
        )

        plt.ylabel(
            "Loss",
            fontsize=12,
        )

        plt.xlabel(
            "Epoch",
            fontsize=12,
        )

        plt.yscale("log")
        plt.grid(linestyle="dashed")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    # Testing
    # Instantiate exactly the same model class that
    # was used during training.
    best_model = model_class(
        input_size=input_size,
        n_classes=n_classes,
        device=device,
        target_length=target_length,
    ).to(device)

    best_model.load_state_dict(torch.load(_CHECKPOINT_PATH))

    return MantisAdapter(model=best_model, classes=classes, device=device, scaler=scaler)