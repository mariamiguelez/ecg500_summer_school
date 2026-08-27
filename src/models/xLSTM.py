import time


import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import plot_PCA

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
)


class XLSTMAdapter:
    # Used to adapt this model to the training pipeline interface
    def __init__(self, model, classes: np.ndarray, device, scaler: StandardScaler | None = None):
        self.model = model
        self.classes = classes
        self.device = device
        self.scaler = scaler

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        if x_tensor.dim() == 2:
            x_tensor = x_tensor.unsqueeze(-1)
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
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
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

class MLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,            # how many input features we have
        d_model: int,
        n_classes: int,             # output size
        num_blocks: int,            # number of layers
        num_heads: int,
        context_length: int,        # number of time steps
        conv1d_kernel_size: int,
        qkv_proj_blocksize: int,    #
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        # define the configurations of the blocks
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, # Local causal temporal context
                    qkv_proj_blocksize=qkv_proj_blocksize, # Feature-group size for Q/K/V projections
                    num_heads=num_heads,
                )
            ),
            context_length=context_length, # Number of time steps
            num_blocks=num_blocks, # Number of mLSTM blocks
            embedding_dim=d_model,
        )

        self.encoder = xLSTMBlockStack(xlstm_config)

        # Classification head
        self.output_projection = nn.Linear(
            d_model*context_length,
            n_classes,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x_embed = self.input_projection(x)  # increase the dimensionality from 6 till the model dimensions

        z = self.encoder(
            x_embed)  # Contextualized representation (latent_sequence) size (bs, tokens, dim of the model)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)

        z_flat = z.flatten(start_dim=1) # (bs, n_timesteps*d_model)

        output = self.output_projection(z_flat) # (bs, n_classes)

        return output


class SLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_classes: int,
        num_blocks: int,
        num_heads: int,
        context_length: int,
        conv1d_kernel_size: int,
    ):
        super().__init__()

        # Project input features to model dimension
        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        # sLSTM configuration
        xlstm_config = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size, # Local causal temporal context
                    backend="vanilla", # sLSTM implementation: vanilla PyTorch or custom CUDA
                ),
            ),
            context_length=context_length, # Number of time steps
            num_blocks=num_blocks, # Number of sLSTM blocks
            embedding_dim=d_model,
        )

        self.encoder = xLSTMBlockStack(xlstm_config)

        # Classification head
        self.output_projection = nn.Linear(
            d_model*context_length,
            n_classes,
        )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x_embed = self.input_projection(x)  # increase the dimensionality from 6 till the model dimensions
        #x_embed_pe = self.positional_encoding(x_embed)  # Add the positional encoder step

        z = self.encoder(x_embed)  # Contextualized representation (latent_sequence) size (bs, tokens, dim of the model)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (B, T, d_model)
        z = self.encoder(x)

        # # Final sequence representation
        # z_last = z[:, -1, :]
        z_flat = z.flatten(start_dim=1) #  # (bs, n_timesteps*d_model)

        output = self.output_projection(z_flat) #(bs, 5)


        return output


class XLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_classes: int,
        num_blocks: int,
        num_heads: int,
        context_length: int,
        conv1d_kernel_size: int,
        qkv_proj_blocksize: int,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        xlstm_config = xLSTMBlockStackConfig(
            # MLSTM block configuration
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size,   # Local causal temporal context
                    qkv_proj_blocksize=qkv_proj_blocksize,  # Feature-group size for Q/K/V projections
                    num_heads=num_heads,
                )
            ),
            # SLSTM block
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,  # Local causal temporal context
                    backend="vanilla",  # sLSTM implementation: vanilla PyTorch or custom CUDA
                ),
            ),

            context_length=context_length,  # Number of time steps
            num_blocks=num_blocks,                   # One mLSTM block + one sLSTM block
            embedding_dim=d_model,
            slstm_at=[1],                   # Block 0: mLSTM, Block 1: sLSTM
        )

        self.encoder = xLSTMBlockStack(xlstm_config)

        # Classification head
        self.output_projection = nn.Linear(
            d_model * context_length,
            n_classes,
        )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x_embed = self.input_projection(x)  # increase the dimensionality from 6 till the model dimensions
        #x_embed_pe = self.positional_encoding(x_embed)  # Add the positional encoder step

        z = self.encoder(
            x_embed)  # Contextualized representation (latent_sequence) size (bs, tokens, dim of the model)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)

        # Flatten all temporal representations
        z_flat = z.flatten(start_dim=1)   # (bs, n_timesteps*d_model)

        # Classification logits
        output = self.output_projection(z_flat)  # (B, n_classes)

        return output



class ECGDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def fit_xlstm(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        loss_weights: np.ndarray = None,
        random_state: int = 42,
        epochs: int = 200,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        use_scaling: bool = False,
        input_size: int = 1,
        d_model: int = 64,
        n_blocks: int = 2,
        n_heads: int = 4,
        conv1d_kernel_size: int = 4,
        qkv_proj_blocksize: int = 4,
        plot_losses: bool = True,
) -> XLSTMAdapter                                                                                 :
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if n_blocks < 2:
        raise ValueError(
            "XLSTMClassifier requires n_blocks >= 2 when using slstm_at=[1]. "
            f"Got n_blocks={n_blocks}."
        )

    if x_val is None or y_val is None:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            stratify=y_train,
            test_size=0.1,
            random_state=random_state,
        )

    train_data, train_labels = x_train, y_train
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
    n_timesteps = train_data.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = XLSTMClassifier(
        input_size=input_size,
        n_classes=len(classes),
        d_model=d_model,
        num_blocks=n_blocks, # set to 2 or more for xLSTM with mLSTM and sLSTM block
        num_heads=n_heads,
        context_length=n_timesteps,
        conv1d_kernel_size=conv1d_kernel_size,  # to how many times steps you want to keep the info in the Q, K
        qkv_proj_blocksize=qkv_proj_blocksize,  # Only for MLSTM, divide the features in groups of x block size
    ).to(device)

    ### Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Instead of MSE the categories don't have an order
    weight_tensor = torch.tensor(loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    early_stopping_patience = 20
    early_stopping_min_delta = 0.0
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = 0
    time_per_epoch = []

    for epoch in range(epochs):
        start_time = time.time()  # Start time per epoch

        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device).unsqueeze(-1)
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
                x_batch = x_batch.to(device).unsqueeze(-1)
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

    ### Model Testing
    best_model = XLSTMClassifier(
        input_size=input_size,
        n_classes=len(classes),
        d_model=d_model,
        num_blocks=n_blocks,
        num_heads=n_heads,
        context_length=n_timesteps,
        conv1d_kernel_size=conv1d_kernel_size,
        qkv_proj_blocksize=qkv_proj_blocksize,  # Only for MLSTM or XLSTM
    ).to(device)

    best_model.load_state_dict(torch.load("models/best_xlstm_model.pth", map_location=device))

    return XLSTMAdapter(model=best_model, classes=classes, device=device, scaler=scaler)

