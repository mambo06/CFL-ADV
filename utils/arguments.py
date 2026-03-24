"""
Command-line argument parser and configuration loader for federated learning experiments.
Reference: https://github.com/AstraZeneca/SubTab
"""

import sys
from argparse import ArgumentParser, Namespace
from typing import Dict, Any

import torch as th

from utils.utils import get_runtime_and_model_config, print_config


# ─── Custom Parser ──────────────────────────────────────────────────────────

class ArgParser(ArgumentParser):
    """
    Custom ArgumentParser that prints helpful messages on error.
    Inherits from ArgumentParser and overrides error handling.
    """
    def error(self, message: str) -> None:
        sys.stderr.write(f"error: {message}\n")
        self.print_help()
        sys.exit(2)


# ─── Argument Definitions ───────────────────────────────────────────────────

def get_arguments() -> Namespace:
    """
    Parse command-line arguments for federated learning experiments.
    
    Returns:
        Namespace containing all parsed arguments
    """
    parser = ArgParser(
        description="Federated Learning with Adversarial Attacks and Defenses"
    )

    # ─── Dataset & Experiment ───────────────────────────────────────────────
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="covtype",
        help="Dataset name (must have a matching config file)"
    )
    
    parser.add_argument(
        "-ex", "--experiment",
        type=int,
        default=1,
        help="Experiment ID suffix for MLFlow tracking"
    )

    # ─── Device Selection ───────────────────────────────────────────────────
    parser.add_argument(
        "-g", "--gpu",
        dest="gpu",
        action="store_true",
        help="Use CUDA GPU (if available)"
    )
    
    parser.add_argument(
        "-ng", "--no_gpu",
        dest="gpu",
        action="store_false",
        help="Force CPU usage"
    )
    
    parser.add_argument(
        "-m", "--mps",
        dest="mps",
        action="store_true",
        help="Use Apple Silicon GPU (if available)"
    )
    
    parser.add_argument(
        "-dn", "--device_number",
        type=str,
        default="0",
        help="CUDA device number (e.g., '0' for cuda:0)"
    )
    
    parser.set_defaults(gpu=True)

    # ─── Training Configuration ─────────────────────────────────────────────
    parser.add_argument(
        "-e", "--epoch",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "-c", "--client",
        type=int,
        default=4,
        help="Number of federated clients"
    )
    
    parser.add_argument(
        "-s", "--sampling",
        type=float,
        default=1.0,
        help="Client sampling ratio per round (0.0-1.0)"
    )
    
    parser.add_argument(
        "-lc", "--local",
        dest="local",
        action="store_true",
        help="Enable local mode (non-federated)"
    )

    # ─── Attack Configuration ───────────────────────────────────────────────
    parser.add_argument(
        "-mc", "--malClient",
        type=float,
        default=0.0,
        help="Fraction of malicious clients (0.0-1.0)"
    )
    
    parser.add_argument(
        "-at", "--attack_type",
        type=str,
        default="scale",
        choices=["scale", "model_replacement", "direction", "gradient_ascent", "targeted"],
        help="Type of adversarial attack"
    )

    # ─── Defense Configuration ──────────────────────────────────────────────
    parser.add_argument(
        "-dt", "--defense_type",
        type=str,
        default="random",
        choices=[
            "multi_krum",
            "geometric_median",
            "foolsgold",
            "trimmed_mean",
            "momentum",
            "random",
            "robust"
        ],
        help="Type of defense mechanism"
    )
    
    parser.add_argument(
        "-rl", "--randomLevel",
        type=float,
        default=0.8,
        help="Randomness level for random defense (0.0-1.0)"
    )

    # ─── Additional Flags ───────────────────────────────────────────────────
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed configuration summary"
    )
    
    parser.add_argument(
        "--save_weights",
        action="store_true",
        default=True,
        help="Save model weights after training"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


# ─── Configuration Builder ──────────────────────────────────────────────────

def get_config(args: Namespace) -> Dict[str, Any]:
    """
    Build complete configuration by merging YAML config with CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Complete configuration dictionary
    """
    # Load base config from YAML files
    config = get_runtime_and_model_config(args)

    # ─── Device Selection ───────────────────────────────────────────────────
    config["device"] = _select_device(args)

    # ─── Training Parameters ────────────────────────────────────────────────
    config.update({
        "local": args.local,
        "epochs": args.epoch,
        "fl_cluster": args.client,
        "sampling": args.sampling,
    })

    # ─── Attack & Defense ───────────────────────────────────────────────────
    config.update({
        "malClient": args.malClient,
        "randomLevel": args.randomLevel,
        "attack_type": args.attack_type,
        "defense_type": args.defense_type,
    })

    # ─── Optional Flags ─────────────────────────────────────────────────────
    config.update({
        "verbose": args.verbose,
        "save_weights": args.save_weights,
    })
    
    if args.seed is not None:
        config["seed"] = args.seed

    return config


def _select_device(args: Namespace) -> th.device:
    """
    Select compute device based on availability and user preference.
    
    Priority:
    1. CUDA GPU (if available and --gpu flag set)
    2. Apple Silicon MPS (if available and --mps flag set)
    3. CPU (fallback)
    
    Args:
        args: Parsed arguments containing device preferences
        
    Returns:
        PyTorch device object
    """
    if th.cuda.is_available() and args.gpu:
        device = th.device(f"cuda:{args.device_number}")
        print(f"[INFO] Using CUDA GPU: {device}")
        return device
    
    if th.backends.mps.is_built() and args.mps:
        device = th.device("mps")
        print("[INFO] Using Apple Silicon GPU (MPS)")
        return device
    
    print("[INFO] Using CPU")
    return th.device("cpu")


# ─── Display Utilities ──────────────────────────────────────────────────────

def print_config_summary(config: Dict[str, Any], args: Namespace | None = None) -> None:
    """
    Print formatted summary of configuration and arguments.
    
    Args:
        config: Configuration dictionary
        args: Optional command-line arguments namespace
    """
    separator = "=" * 100
    
    print(separator)
    print("Configuration Summary\n")
    print_config(config)
    print(separator)
    
    if args is not None:
        print("\nCommand-line Arguments\n")
        print_config(vars(args))
        print(separator)


# ─── Validation ─────────────────────────────────────────────────────────────

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values and raise errors for invalid settings.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration contains invalid values
    """
    # Validate ranges
    if not 0.0 <= config["malClient"] <= 1.0:
        raise ValueError(f"malClient must be in [0.0, 1.0], got {config['malClient']}")
    
    if not 0.0 <= config["randomLevel"] <= 1.0:
        raise ValueError(f"randomLevel must be in [0.0, 1.0], got {config['randomLevel']}")
    
    if not 0.0 <= config["sampling"] <= 1.0:
        raise ValueError(f"sampling must be in [0.0, 1.0], got {config['sampling']}")
    
    if config["epochs"] <= 0:
        raise ValueError(f"epochs must be positive, got {config['epochs']}")
    
    if config["fl_cluster"] <= 0:
        raise ValueError(f"fl_cluster must be positive, got {config['fl_cluster']}")
    
    # Validate attack/defense compatibility (optional)
    if config["malClient"] == 0.0 and config["defense_type"] != "random":
        print("[WARNING] No malicious clients but defense is active")


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test argument parsing and configuration building."""
    args = get_arguments()
    config = get_config(args)
    
    try:
        validate_config(config)
        print_config_summary(config, args)
    except ValueError as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        sys.exit(1)