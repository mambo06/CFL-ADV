import yaml
import json
import copy
import random
from pathlib import Path
from tqdm import tqdm

import evaluation as eval
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.utils import set_dirs, set_seed, run_with_profiler, update_config_with_model_dims
from utils.load_data import Loader
from src.model import CFL
from attacks.attackmanager import AttackManager
from clients.maliciousclient import MaliciousClient
from clients.secureclient import SecureClient
from servers.robustserver import RobustServer


# ─── Helpers ────────────────────────────────────────────────────────────────

def build_experiment_tag(config: dict, client_id: int | None = None) -> str:
    """Build a consistent experiment identifier string."""
    prefix = f"Cl-{client_id}-" if client_id is not None else ""
    return (
        f"{prefix}"
        f"{config['epochs']}e-"
        f"{config['fl_cluster']}fl-"
        f"{config['malClient']}mc-"
        f"{config['attack_type']}_at-"
        f"{config['defense_type']}_dt-"
        f"{config['randomLevel']}rl-"
        f"{config['dataset']}"
    )


# ─── Run ────────────────────────────────────────────────────────────────────

def run(config: dict, save_weights: bool = True) -> None:
    config = copy.deepcopy(config)
    config["client"] = 0

    # Data & model setup
    ds_loader = Loader(config, dataset_name=config["dataset"])
    config = update_config_with_model_dims(ds_loader, config)
    global_model = CFL(config)

    server = RobustServer(model=global_model, config=config)
    attack_manager = AttackManager(config)

    # Select poisoned clients
    num_malicious = int(config["fl_cluster"] * config["malClient"])
    poison_clients = (
        set(random.sample(range(config["fl_cluster"]), num_malicious))
        if num_malicious > 0
        else set()
    )

    print(f"[INFO] Poisoned clients : {sorted(poison_clients)}")
    print(f"[INFO] Attack type      : {attack_manager.attack_type.value}")
    print(f"[INFO] Defense type     : {config['defense_type']}")
    print(f"[INFO] Experiment tag   : {build_experiment_tag(config)}")

    # Build clients
    clients = []
    for clt in range(config["fl_cluster"]):
        config["prefix"] = build_experiment_tag(config, client_id=clt)
        config["client"] = clt
        loader = Loader(config, dataset_name=config["dataset"]).trainFL_loader

        if clt in poison_clients:
            client = MaliciousClient(
                model=global_model,
                dataloader=loader,
                client_number=clt,
                config=config,
                attack_manager=attack_manager,
            )
        else:
            client = SecureClient(
                model=global_model,
                dataloader=loader,
                client_number=clt,
                config=config,
            )

        client.poison = clt in poison_clients
        clients.append(client)

    # Training loop
    total_batches = len(loader)
    for epoch in range(config["epochs"]):
        tloss = 0.0
        for _ in tqdm(range(total_batches), desc=f"Epoch {epoch+1}/{config['epochs']}"):
            for client in clients:
                tloss += client.train().item()

            server.aggregate(client_models=clients)

            for client in clients:
                client.set_model(server.distribute_model())
                client.model.loss["tloss_e"].append(
                    sum(client.model.loss["tloss_b"][-total_batches:]) / total_batches
                )

        avg_loss = tloss / (config["fl_cluster"] * total_batches)
        print(f"[Epoch {epoch+1:>3}] avg loss: {avg_loss:.4f}")

    # Persist results
    for n, client in enumerate(clients):
        model = client.model
        model.saveTrainParams(n)

        if save_weights:
            model.save_weights(n)

        tag = build_experiment_tag(config, client_id=n)
        config_out = Path(model._results_path) / f"config_{tag}.yml"
        config_out.write_text(yaml.dump(config, default_flow_style=False))


# ─── Main ───────────────────────────────────────────────────────────────────

def main(config: dict) -> None:
    config["framework"] = config["dataset"]

    info = json.loads(Path(f'data/{config["dataset"]}/info.json').read_text())

    # Dataset metadata
    config.update({
        "task_type": info["task_type"],
        "cat_policy": info["cat_policy"],
        "norm": info["norm"],
    })

    # Attack settings (overridable via CLI)
    config.setdefault("attack_probability", 1.0)
    config.setdefault("target_layer", "encoder.layer1")
    config.setdefault("noise_std", 0.1)
    config.setdefault("learning_rate_reducer", config["learning_rate"])

    # Defense hyper-parameters (overridable via CLI)
    config.setdefault("trim_ratio", 0.1)
    config.setdefault("random_level", 0.8)
    config.setdefault("history_size", 10)
    config.setdefault("num_groups", 5)
    config.setdefault("eps", 0.5)
    config.setdefault("min_samples", 3)
    config.setdefault("num_subsets", 5)
    config.setdefault("subset_size", 0.8)
    config.setdefault("window_size", 10)
    config.setdefault("detection_threshold", 2.0)

    if config.get("verbose"):
        print_config_summary(config)

    run(copy.deepcopy(config), save_weights=config.get("save_weights", True))
    eval.main(copy.deepcopy(config))


if __name__ == "__main__":
    args = get_arguments()
    config = get_config(args)
    main(config)