import numpy as np
import time
import torch
import torch.nn as nn
import pywt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import plot_PCA


class EncoderAdapter:
    # Used to adapt this model to the training pipeline interface
    def __init__(
        self,
        model,
        classes: np.ndarray,
        device,
        scales: np.ndarray,
        wavelet: str,
        use_wavelet: bool,
        scaler: StandardScaler | None = None,
    ):
        self.model = model
        self.classes = classes
        self.device = device
        self.scales = scales
        self.wavelet = wavelet
        self.use_wavelet = use_wavelet
        self.scaler = scaler
        self.handles_input_scaling = True

    def _transform_batch(self, x: np.ndarray) -> np.ndarray:
        if not self.use_wavelet:
            return x[..., None].astype(np.float32)

        transformed = []
        for series in x:
            coefficients, _ = pywt.cwt(series, self.scales, self.wavelet)
            transformed.append(
                np.stack([coefficients.real, coefficients.imag], axis=-1)
                .transpose(1, 0, 2)
                .reshape(len(series), -1)
                .astype(np.float32)
            )
        return np.stack(transformed, axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        wavelet_data = self._transform_batch(x)
        if self.scaler is not None:
            wavelet_data = self.scaler.transform(
                wavelet_data.reshape(-1, wavelet_data.shape[-1])
            ).reshape(wavelet_data.shape)
        x_tensor = torch.tensor(wavelet_data, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x_tensor) #obtain logits from the linear prediction which can take any value in R
            # get the indice with the highest logit value as after transforming to back to
            # probabilities the same value will have the highest prob
            class_idx = torch.argmax(logits, dim=1).cpu().numpy()

        return self.classes[class_idx] # return the class taking the indice with the highes logit

    def extract_latent_vectors(
        self,
        x: np.ndarray,
        pooling: str = "mean",
    ) -> np.ndarray:
        """Extract sequence-level latent vectors from the transformer encoder."""
        self.model.eval()
        wavelet_data = self._transform_batch(x)
        if self.scaler is not None:
            wavelet_data = self.scaler.transform(
                wavelet_data.reshape(-1, wavelet_data.shape[-1])
            ).reshape(wavelet_data.shape)
        x_tensor = torch.tensor(wavelet_data, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self.model.encode(x_tensor)  # (B, T, d_model)

            if pooling == "mean":
                latents = z.mean(dim=1)
            elif pooling == "last":
                latents = z[:, -1, :]
            elif pooling == "flatten":
                latents = z.flatten(start_dim=1)

        return latents.cpu().numpy()

    def plot_latent_pca(
        self,
        x: np.ndarray,
        labels: np.ndarray | None = None,
        pooling: str = "mean",
        output_path: str | None = None,
        title: str = "2D Projection of Transformer Latent Space",
    ) -> np.ndarray:
        """Extract latent vectors and plot them using PCA."""
        latent_vectors = self.extract_latent_vectors(x=x, pooling=pooling)
        return plot_PCA(
            latent_vectors=latent_vectors,
            labels=labels,
            output_path=output_path,
            title=title,
        )

class BaseTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        dim_ff: int,        # Dim of the feedforward nn
        n_classes: int,
        num_layers: int,
        n_heads: int,
        n_tokens: int,      # aka number of time steps
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)  # From 6 to the model timension

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=n_tokens)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers) # Num of layers how many stacked encoder layers you want

        self.output_projection = nn.Linear(d_model * n_tokens, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x_embed = self.input_projection(x) # increase the dimensionality from 6 till the model dimensions
        x_embed_pe = self.positional_encoding(x_embed) # Add the positional encoder step

        z = self.encoder(x_embed_pe)    # Contextualized representation (latent_sequence) size (bs, tokens, dim of the model)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        # The output projecton has the size of (bs, tokens), therefore should be flatten

        # Flatten the output of the encoder
        z_flat = z.flatten(start_dim=1)

        logits = self.output_projection(z_flat) # (bs, n_classes)

        return logits


class CLSTransformerClassifier(nn.Module):
    # Compress into 4 neuns,By aggregating everything we can pass it through the linear projection layer
    def __init__(
        self,
        input_size,
        n_classes,
        d_model,
        n_heads,
        num_layers,
        dim_ff,
        n_tokens,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=n_tokens + 1 # this +1 is the cls token
        )
        self.cls_token = nn.Parameter(torch.randn(1, d_model)) # randn is the learnable parameter that can use

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, F_in)
        mask: (B, T) bool (True = real, False = pad)
        """
        x = self.input_proj(x)
        bs = x.shape[0]

        cls_tokens = self.cls_token.expand(bs, 1, -1) # Expanded to include the batch size
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.positional_encoding(x)


        x = self.encoder(x) # (bs, tokens +cls, dim_model)
        summary = x[:, 0, :] # extract the cls token (bs, dim_model)

        logits = self.output_projection(summary)

        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class LearnedPositionalEmbedding(nn.Module):
    """Learned Positional Embedding.
    Args:
        max_len (int): Maximum length of the input sequences.
        d_model (int): Dimension of the model.
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model) # trainable matrix size of the max time steps
        self.register_buffer("positions", torch.arange(0, max_len).long())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """

        seq_len = x.size(1) # how many tokens we have
        positions = self.positions[:seq_len]
        pos_embed = self.pos_embedding(positions).unsqueeze(0)
        x = x + pos_embed
        return x


class ECGDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def fit_transformer(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        random_state: int = 42,
        loss_weights: np.ndarray = None,
        epochs: int = 200,
        input_size: int | None = None,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        use_scaling: bool = False,
        d_model: int = 64,
        dim_ff: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        use_wavelet: bool | None = None,
        wavelet: str = "morl",
        max_scale: int = 64,
        plot_losses: bool = True, # Option to plot loss

) -> EncoderAdapter:
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

    scales = np.arange(1, max_scale + 1)
    if use_wavelet is None:
        use_wavelet = input_size == 2 * len(scales)
    input_size = 2 * len(scales) if use_wavelet else 1

    def transform_batch(x: np.ndarray) -> np.ndarray:
        transformed = []
        for series in x:
            coefficients, _ = pywt.cwt(series, scales, wavelet)
            transformed.append(
                np.stack([coefficients.real, coefficients.imag], axis=-1)
                .transpose(1, 0, 2)
                .reshape(len(series), -1)
                .astype(np.float32)
            )
        return np.stack(transformed, axis=0)

    if use_wavelet:
        train_data, train_labels = transform_batch(x_train), y_train
        x_val = transform_batch(x_val)
    else:
        train_data, train_labels = x_train[..., None].astype(np.float32), y_train
        x_val = x_val[..., None].astype(np.float32)
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
    n_classes = len(classes)
    n_tokens = train_data.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaseTransformerClassifier(
        input_size=input_size,
        n_classes=n_classes,
        d_model=d_model,
        dim_ff=dim_ff,
        n_heads=n_heads,
        num_layers=n_layers,
        n_tokens=n_tokens,
    ).to(device)

    ### Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Instead of MSE the categories don't have an order
    weight_tensor = torch.tensor(loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = 0
    early_stopping_patience = 20
    early_stopping_min_delta = 0.0
    epochs_without_improvement = 0
    time_per_epoch = []

    for epoch in range(epochs):
        start_time = time.time()  # Start time per epoch

        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
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
        train_losses.append(train_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            # Pass each batch from the validation loader into the output
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(x_batch)

                loss = criterion(predictions, y_batch)

                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
        end_time = time.time()
        time_per_epoch.append(end_time - start_time)  # Appends the time per epoch

        if val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = val_loss
            best_val_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "models/best_endoder_model.pth")
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.8f}"
            f"| Val loss: {val_loss:.8f}"
            f" | "
            f"Best Val loss: {best_val_loss:.8f}"
            f" | "
            f"Best Val epoch: {best_val_epoch + 1}"
        )

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"no validation improvement for {early_stopping_patience} epochs."
            )
            break
    average_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
    print(f"Average time per epoch: {average_time_per_epoch:.4f}s")

    ### Plot the loss function
    if plot_losses:
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.ylabel("Loss", fontsize=12)
        plt.xlabel("Epoch", fontsize=12)
        plt.yscale("log")
        plt.grid(linestyle="dashed")
        plt.legend()
        plt.show()
        plt.close()


    best_model = BaseTransformerClassifier(
        input_size=input_size,
        n_classes=n_classes,
        d_model=d_model,
        dim_ff=dim_ff,
        n_heads=n_heads,
        num_layers=n_layers,
        n_tokens=n_tokens,
    ).to(device)
    best_model.load_state_dict(torch.load("models/best_endoder_model.pth", map_location=device))

    return EncoderAdapter(
        model=best_model,
        classes=classes,
        device=device,
        scales=scales,
        wavelet=wavelet,
        use_wavelet=use_wavelet,
        scaler=scaler,
    )