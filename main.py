"""
Run this file, giving a configuration file as input, to train models, e.g.:
        python main.py --config_name configfile.yaml
"""

from os.path import join
from pathlib import Path
import wandb
import torch
import torch.optim as optim
import hydra
from omegaconf import DictConfig

from models.losses import create_loss
from utils.general import reset_random_seeds, check_device, set_paths, init_wandb
from utils.data import get_data
from utils.models import create_model
from utils.training import (
    train_one_epoch,
    validate_one_epoch,
    evaluate_model,
    create_optimizer,
    Custom_Metrics,
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """
    Train and test routine
    """
    # ---------------------------------
    #       Setup
    # ---------------------------------
    # Reproducibility
    gen = reset_random_seeds(config.seed)
    device = check_device()

    # Set paths
    config = set_paths(config)

    # Print configuration
    print("Project directory:", Path(__file__).absolute().parent)
    print("Configuration:", config)
    print("Experiment path: ", config.experiment_path)

    # Wandb
    init_wandb(config)

    # ----------------------------------------------
    #       Prepare training
    # ----------------------------------------------
    # Load data
    train_loader, val_loader, test_loader = get_data(config.data, gen)

    # Initialize model & loss
    model = create_model(config)
    model.to(device)
    loss_fn = create_loss(config)

    # Initialize other objects needed for training
    optimizer = create_optimizer(config.model, model)
    metrics = Custom_Metrics(device).to(device)

    print("STARTING TRAINING " + str(config.model.model))

    for epoch in range(0, config.model.num_epochs):
        if epoch % config.model.validate_per_epoch == 0:
            print("\nEVALUATION ON THE VALIDATION SET:\n")
            validate_one_epoch(val_loader, model, metrics, config, loss_fn, device)
        train_one_epoch(
            train_loader, model, optimizer, metrics, epoch, config, loss_fn, device
        )

    model.eval()
    if config.save_model:
        torch.save(model.state_dict(), join(config.experiment_path, "model.pth"))
        print("\nTRAINING FINISHED, MODEL SAVED!", flush=True)
    else:
        print("\nTRAINING FINISHED", flush=True)

    if not config.model.use_dynamic_threshold:
        evaluate_model(test_loader, model, metrics, config, loss_fn, device)
    else:
        # Evaluate the model at different certainty thresholds
        for threshold in config.model.certainty_threshold:
            model.selector.certainty_threshold = threshold
            evaluate_model(test_loader, model, metrics, config, loss_fn, device)

    wandb.finish(quiet=True)
    return None


if __name__ == "__main__":
    main()
