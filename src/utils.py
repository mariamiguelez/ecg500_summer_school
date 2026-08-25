import numpy as np
from pathlib import Path
from typing import Any
import yaml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}

def plot_PCA(
    latent_vectors: np.ndarray,
    labels: np.ndarray | None = None,
    output_path: str | Path | None = None,
    title: str = "2D Projection of Latent Space",
) -> np.ndarray:
    """Project latent vectors to 2D with PCA and plot them."""
    if latent_vectors.ndim != 2:
        raise ValueError(
            f"Expected latent_vectors with shape (n_samples, n_features). "
            f"Got shape {latent_vectors.shape}."
        )

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 8))
    if labels is None:
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
    else:
        scatter = plt.scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Classes")

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(linestyle="dashed")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    return latent_2d