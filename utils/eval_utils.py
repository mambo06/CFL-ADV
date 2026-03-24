"""
Evaluation utilities for federated contrastive learning.
Includes model evaluation, visualization, and result aggregation functions.
Reference: https://github.com/AstraZeneca/SubTab
"""

import csv
import functools
import os
from typing import Dict, Any, List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from utils.utils import tsne
from utils.colors import get_color_list


# ─── Model Evaluation ───────────────────────────────────────────────────────

def linear_model_eval(
    config: Dict[str, Any],
    z_train: np.ndarray,
    y_train: np.ndarray,
    suffix: str,
    z_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    description: str = "Logistic Regression",
    nData: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate representations using Logistic Regression.
    
    Args:
        config: Configuration dictionary
        z_train: Training embeddings (N × D)
        y_train: Training labels (N,)
        suffix: Suffix for result filename
        z_test: Test embeddings (M × D)
        y_test: Test labels (M,)
        description: Description of evaluation
        nData: Dataset identifier (optional)
        
    Returns:
        List of result dictionaries containing train/test metrics
    """
    results_list = []
    
    # Build filename
    file_name = suffix + config["prefix"][5:]
    
    # Define regularization parameters to sweep
    if nData is None:
        regularisation_list = [0.0001, 1, 10]
    else:
        regularisation_list = [0.001]
    
    # Override with single value
    regularisation_list = [10]
    
    # Evaluate for each regularization parameter
    for c in regularisation_list:
        # Initialize and train logistic regression
        clf = LogisticRegression(max_iter=1200, solver="lbfgs", C=c)
        clf.fit(z_train, y_train)
        
        # Make predictions
        y_hat_train = clf.predict(z_train)
        y_hat_test = clf.predict(z_test)
        
        # Compute metrics
        tr_metrics = precision_recall_fscore_support(
            y_train, y_hat_train, average="weighted"
        )
        te_metrics = precision_recall_fscore_support(
            y_test, y_hat_test, average="weighted"
        )
        
        # Print results
        print(f"\n[INFO] Logistic Regression (C={c})")
        print(f"Training  - Precision: {tr_metrics[0]:.4f}, "
              f"Recall: {tr_metrics[1]:.4f}, "
              f"F1: {tr_metrics[2]:.4f}")
        print(f"Test      - Precision: {te_metrics[0]:.4f}, "
              f"Recall: {te_metrics[1]:.4f}, "
              f"F1: {te_metrics[2]:.4f}")
        
        # Record results
        results_list.append({
            "model": f"LogReg_{c}",
            "train_precision": tr_metrics[0],
            "train_recall": tr_metrics[1],
            "train_f1": tr_metrics[2],
            "test_precision": te_metrics[0],
            "test_recall": te_metrics[1],
            "test_f1": te_metrics[2],
        })
    
    # Save results to CSV
    _save_results_to_csv(config, results_list, file_name)
    
    return results_list


def _save_results_to_csv(
    config: Dict[str, Any],
    results_list: List[Dict[str, Any]],
    file_name: str
) -> None:
    """
    Save evaluation results to CSV file.
    
    Args:
        config: Configuration dictionary
        results_list: List of result dictionaries
        file_name: Base filename for CSV
    """
    if not results_list:
        print("[WARNING] No results to save")
        return
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join("results", config["dataset"])
    os.makedirs(results_dir, exist_ok=True)
    
    # Build file path
    file_path = os.path.join(results_dir, f"{file_name}.csv")
    
    # Write to CSV
    keys = results_list[0].keys()
    with open(file_path, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_list)
    
    print(f"[INFO] Results saved to: {file_path}")


# ─── Visualization ──────────────────────────────────────────────────────────

def plot_clusters(
    config: Dict[str, Any],
    z: np.ndarray,
    clabels: np.ndarray,
    suffix: str,
    plot_suffix: str = "_inLatentSpace"
) -> None:
    """
    Wrapper function to visualize clusters using PCA and t-SNE.
    
    Args:
        config: Configuration dictionary
        z: Embeddings (N × D)
        clabels: Class labels (N,)
        suffix: Suffix for plot filename
        plot_suffix: Additional suffix for plot name
    """
    # Get number of unique classes
    n_classes = len(np.unique(clabels))
    
    # Generate legend labels
    clegends = [str(i) for i in range(n_classes)]
    
    # Visualize clusters
    visualise_clusters(
        config, z, clabels, suffix,
        plt_name=f"classes{plot_suffix}",
        legend_title="Classes",
        legend_labels=clegends
    )


def visualise_clusters(
    config: Dict[str, Any],
    embeddings: np.ndarray,
    labels: np.ndarray,
    suffix: str,
    plt_name: str = "test",
    alpha: float = 1.0,
    legend_title: Optional[str] = None,
    legend_labels: Optional[List[str]] = None,
    ncol: int = 1
) -> None:
    """
    Visualize clusters using PCA and t-SNE projections.
    
    Args:
        config: Configuration dictionary
        embeddings: Embeddings to visualize (N × D)
        labels: Class labels (N,)
        suffix: Suffix for plot filename
        plt_name: Name for the plot file
        alpha: Transparency of scatter points
        legend_title: Title for legend
        legend_labels: Labels for legend
        ncol: Number of columns in legend
    """
    # Setup color palette
    color_list, _ = get_color_list()
    palette = {str(i): color_list[i] for i in range(len(color_list))}
    
    # Prepare labels
    y = labels.reshape(-1).astype(str).tolist()
    
    # Create subplots
    fig, axs = plt.subplots(
        1, 2,
        figsize=(9, 3.5),
        facecolor="w",
        edgecolor="k"
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    # Ensure axs is always a list
    axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs, axs]
    
    # ─── PCA Projection ───
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    axs[0].set_title("Embeddings from PCA")
    sns.scatterplot(
        x=embeddings_pca[:, 0],
        y=embeddings_pca[:, 1],
        ax=axs[0],
        palette=palette,
        hue=y,
        s=20,
        alpha=alpha,
        legend=False
    )
    _configure_axis(axs[0])
    
    # ─── t-SNE Projection ───
    embeddings_tsne = tsne(embeddings)
    
    axs[1].set_title("Embeddings from t-SNE")
    sns_plt = sns.scatterplot(
        x=embeddings_tsne[:, 0],
        y=embeddings_tsne[:, 1],
        ax=axs[1],
        palette=palette,
        hue=y,
        s=20,
        alpha=alpha
    )
    _configure_axis(axs[1])
    
    # Configure legend
    _setup_legend(
        sns_plt, fig,
        ncol=ncol,
        labels=legend_labels,
        title=legend_title or "Cluster"
    )
    
    # Remove individual legends
    for ax in axs:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    
    # Adjust layout for legend
    legend_space = {1: 0.9, 2: 0.9, 3: 0.75, 4: 0.65, 5: 0.65}
    plt.subplots_adjust(right=legend_space.get(ncol, 0.9))
    
    # Save plot
    _save_plot(config, suffix, plt_name)
    
    plt.close(fig)


def _configure_axis(ax: plt.Axes) -> None:
    """
    Configure axis appearance for cluster plots.
    
    Args:
        ax: Matplotlib axis object
    """
    ax.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    ax.tick_params(
        top=False, bottom=False,
        left=False, right=False,
        labeltop=False, labelbottom=False,
        labelleft=False, labelright=False
    )


def _setup_legend(
    sns_plt: plt.Axes,
    fig: plt.Figure,
    ncol: int,
    labels: Optional[List[str]],
    title: str
) -> None:
    """
    Setup legend for cluster plots.
    
    Args:
        sns_plt: Seaborn plot object
        fig: Figure object
        ncol: Number of legend columns
        labels: Legend labels
        title: Legend title
    """
    # Get handles and labels
    handles, legend_txts = sns_plt.get_legend_handles_labels()
    
    # Sort by label
    legend_txts = [int(d) for d in legend_txts]
    legend_txts, handles = zip(*sorted(zip(legend_txts, handles)))
    
    # Use provided labels or default
    display_labels = labels or [str(i) for i in range(len(handles))]
    
    # Create legend
    fig.legend(
        handles,
        display_labels,
        loc="center right",
        borderaxespad=0.1,
        title=title,
        ncol=ncol
    )


def _save_plot(
    config: Dict[str, Any],
    suffix: str,
    plt_name: str
) -> None:
    """
    Save plot to file.
    
    Args:
        config: Configuration dictionary
        suffix: Filename suffix
        plt_name: Plot name
    """
    # Create directory structure
    root_path = os.path.dirname(os.path.dirname(__file__))
    fig_dir = os.path.join(
        root_path, "results", config["framework"],
        "evaluation", "clusters"
    )
    os.makedirs(fig_dir, exist_ok=True)
    
    # Build file path
    fig_path = os.path.join(fig_dir, f"{suffix}{plt_name}.png")
    
    # Save figure
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"[INFO] Plot saved to: {fig_path}")


# ─── Data Utilities ─────────────────────────────────────────────────────────

def append_tensors_to_lists(
    list_of_lists: List[List],
    list_of_tensors: List[th.Tensor]
) -> List[List]:
    """
    Append tensors to lists after converting to numpy arrays.
    
    Args:
        list_of_lists: List of lists to append to
        list_of_tensors: List of tensors to convert and append
        
    Returns:
        Updated list of lists
    """
    for i, tensor in enumerate(list_of_tensors):
        list_of_lists[i].append(tensor.cpu().numpy())
    
    return list_of_lists


def concatenate_lists(list_of_lists: List[List]) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Concatenate lists of arrays into single arrays.
    
    Args:
        list_of_lists: List of lists containing numpy arrays
        
    Returns:
        Single numpy array or list of numpy arrays
    """
    list_of_np_arrs = [np.concatenate(list_) for list_ in list_of_lists]
    
    return list_of_np_arrs[0] if len(list_of_np_arrs) == 1 else list_of_np_arrs


def save_np2csv(
    np_list: List[np.ndarray],
    save_as: str = "test.csv"
) -> None:
    """
    Save numpy arrays to CSV file.
    
    Args:
        np_list: List containing [features, labels]
        save_as: Filename for CSV
    """
    # Extract features and labels
    X, y = np_list
    y = np.array(y, dtype=np.int8)
    
    # Create column names
    columns = ["label"] + [str(i) for i in range(X.shape[1])]
    
    # Concatenate features and labels
    data = np.concatenate((y.reshape(-1, 1), X), axis=1)
    
    # Create dataframe
    df = pd.DataFrame(data=data, columns=columns)
    
    # Save to CSV
    df.to_csv(save_as, index=False)
    
    print(f"[INFO] Dataframe saved to: {save_as}")
    print(f"[INFO] Sample:\n{df.head()}")


# ─── Aggregation Functions ──────────────────────────────────────────────────

def aggregate(
    latent_list: List[th.Tensor],
    config: Dict[str, Any]
) -> th.Tensor:
    """
    Aggregate latent representations from multiple subsets.
    
    Args:
        latent_list: List of latent tensors from different subsets
        config: Configuration dictionary with 'aggregation' key
        
    Returns:
        Aggregated latent representation
    """
    aggregation_method = config.get("aggregation", "mean")
    
    if aggregation_method == "mean":
        return th.mean(th.stack(latent_list), dim=0)
    
    elif aggregation_method == "sum":
        return th.sum(th.stack(latent_list), dim=0)
    
    elif aggregation_method == "concat":
        return th.cat(latent_list, dim=-1)
    
    elif aggregation_method == "max":
        return th.max(th.stack(latent_list), dim=0)[0]
    
    elif aggregation_method == "min":
        return th.min(th.stack(latent_list), dim=0)[0]
    
    else:
        raise ValueError(
            f"Unknown aggregation method: {aggregation_method}. "
            f"Choose from: mean, sum, concat, max, min"
        )


# ─── Advanced Aggregation ───────────────────────────────────────────────────

class WeightedAggregation:
    """Weighted aggregation of latent representations."""
    
    def __init__(self, n_subsets: int, method: str = "learned"):
        """
        Initialize weighted aggregation.
        
        Args:
            n_subsets: Number of subsets to aggregate
            method: Weighting method ('uniform', 'learned', 'attention')
        """
        self.n_subsets = n_subsets
        self.method = method
        
        if method == "uniform":
            self.weights = th.ones(n_subsets) / n_subsets
        elif method == "learned":
            self.weights = th.nn.Parameter(th.ones(n_subsets) / n_subsets)
        elif method == "attention":
            # Attention mechanism would be implemented here
            self.weights = th.ones(n_subsets) / n_subsets
        else:
            raise ValueError(f"Unknown weighting method: {method}")
    
    def __call__(self, latent_list: List[th.Tensor]) -> th.Tensor:
        """
        Aggregate latent representations with weights.
        
        Args:
            latent_list: List of latent tensors
            
        Returns:
            Weighted aggregated tensor
        """
        # Stack tensors
        stacked = th.stack(latent_list, dim=0)  # (n_subsets, batch, dim)
        
        # Apply weights
        weights = self.weights.view(-1, 1, 1)  # (n_subsets, 1, 1)
        weighted = stacked * weights
        
        # Sum over subsets
        return th.sum(weighted, dim=0)


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted"
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average
    )
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(support) if average is None else float(support)
    }


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for print statement
    """
    print(f"{prefix}Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, "
          f"F1: {metrics['f1']:.4f}")