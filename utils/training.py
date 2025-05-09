import numpy as np
from torchmetrics import Metric
import torch
from torch import nn
import wandb
from tqdm import tqdm

from utils.evaluation import (
    calc_target_metrics,
    log_images,
    compute_fidelity,
    compute_localization,
)
from utils.general import reset_random_seeds


def evaluate_model(test_loader, model, metrics, config, loss_fn, device):
    """Compute metrics on the test set"""
    reset_random_seeds(config.seed)  # To log same image for each threshold
    print("\nEVALUATION ON THE TEST SET:\n")
    validate_one_epoch(test_loader, model, metrics, config, loss_fn, device, test=True)

    print("\nEVALUATING DELETION FIDELITY BY MASKING HIGHEST ATTRIBUTION REGIONS:\n")
    compute_fidelity(test_loader, model, config, device)
    print("\nEVALUATING INSERTION FIDELITY BY MASKING LOWEST ATTRIBUTION REGIONS:\n")
    compute_fidelity(test_loader, model, config, device, inverse=True)

    if hasattr(test_loader.dataset, "has_segmentation"):
        if test_loader.dataset.has_segmentation:
            print("\nEVALUATING LOCALIZATION:\n")
            compute_localization(test_loader, model, device, config)


def train_one_epoch(
    train_loader, model, optimizer, metrics, epoch, config, loss_fn, device
):
    """
    Train for one epoch
    """

    model.selector.compute_temperature(epoch)
    loss_fn.compute_reg_weight(epoch + 1)
    metrics.reset()

    model.train()
    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        x, groups, target_true = (
            batch[0][0].to(device),
            batch[0][1].to(device),
            batch[1].to(device),
        )

        # Forward pass
        target_pred_logits, groups_probs, pixel_probs, groups_cov = model(x, groups)

        # Compute the loss
        target_loss, reg_loss, cov_loss, total_loss, avg_weighted_group_probs = loss_fn(
            target_pred_logits,
            target_true,
            groups,
            groups_probs,
            groups_cov,
            model,
        )
        # Perform an update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Store predictions
        metrics.update(
            target_loss,
            reg_loss,
            total_loss,
            target_true,
            target_pred_logits,
            avg_weighted_group_probs.mean(),
            pixel_probs.mean(),
            cov_loss,
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    return


def validate_one_epoch(
    loader,
    model,
    metrics,
    config,
    loss_fn,
    device,
    test=False,
):
    """
    Validating one epoch
    """
    model.eval()
    metrics.reset()

    # Randomly log a few images
    log_idx = np.random.choice(len(loader), 10, replace=False)
    with torch.no_grad():

        for k, batch in enumerate(tqdm(loader, position=0, leave=True)):
            x, groups, target_true = (
                batch[0][0].to(device),
                batch[0][1].to(device),
                batch[1].to(device),
            )

            (
                target_pred_logits,
                groups_probs,
                pixel_probs,
                masked_x,
                groups_cov,
            ) = model(x, groups, validation=True)

            target_loss, reg_loss, cov_loss, total_loss, avg_weighted_group_probs = (
                loss_fn(
                    target_pred_logits,
                    target_true,
                    groups,
                    groups_probs,
                    groups_cov,
                    model,
                )
            )

            metrics.update(
                target_loss,
                reg_loss,
                total_loss,
                target_true,
                target_pred_logits,
                avg_weighted_group_probs.mean(),
                pixel_probs.mean(),
                cov_loss,
            )
            if k in log_idx:
                if test:
                    n_imgs = 5
                else:
                    n_imgs = 1
                for _ in range(n_imgs):
                    log_images(
                        x,
                        masked_x,
                        groups,
                        groups_cov,
                        pixel_probs,
                        groups_probs,
                        target_pred_logits,
                        target_true,
                        model,
                        config,
                    )

    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)

    if not test:
        wandb_name = "validation"
        prints = f"Validation: "
    else:
        wandb_name = "test"
        prints = f"Test: "
    if config.model.use_dynamic_threshold:
        wandb_name += f"_threshold_{model.selector.certainty_threshold}"
        prints += f" Certainty Threshold: {model.selector.certainty_threshold} "

    wandb.log({f"{wandb_name}/{k}": v for k, v in metrics_dict.items()})
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    return


class Custom_Metrics(Metric):
    """Lightning-like class to log variables, losses and metrics"""

    def __init__(self, device):
        super().__init__()
        self.add_state("target_loss", default=torch.tensor(0.0, device=device))
        self.add_state("regularization_loss", default=torch.tensor(0.0, device=device))
        self.add_state("total_loss", default=torch.tensor(0.0, device=device))
        self.add_state("y_true", default=[])
        self.add_state("y_pred_logits", default=[])
        self.add_state("cov_loss", default=torch.tensor(0.0, device=device))
        self.add_state("pixel_probs", default=torch.tensor(0.0, device=device))
        self.add_state(
            "avg_weighted_group_probs", default=torch.tensor(0.0, device=device)
        )
        self.add_state(
            "n_samples", default=torch.tensor(0, dtype=torch.int, device=device)
        )

    def update(
        self,
        target_loss: torch.Tensor,
        reg_loss: torch.Tensor,
        total_loss: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_logits: torch.Tensor,
        avg_weighted_group_probs: torch.Tensor,
        pixel_probs: torch.Tensor = None,
        cov_loss: torch.Tensor = None,
    ):
        n_samples = y_true.size(0)
        self.n_samples += n_samples
        self.target_loss += target_loss * n_samples
        self.regularization_loss += reg_loss * n_samples
        self.total_loss += total_loss * n_samples

        self.avg_weighted_group_probs += avg_weighted_group_probs * n_samples
        if pixel_probs:
            self.pixel_probs += pixel_probs * n_samples
        if cov_loss:
            self.cov_loss += cov_loss * n_samples
        self.y_true.append(y_true)
        self.y_pred_logits.append(y_pred_logits.detach())

    def compute(self, validation=False, config=None):
        self.y_true = torch.cat(self.y_true, dim=0).cpu()
        self.y_pred_logits = torch.cat(self.y_pred_logits, dim=0).cpu()

        y_pred_probs = nn.Softmax(dim=1)(self.y_pred_logits)
        y_pred = self.y_pred_logits.argmax(dim=-1)

        target_acc = (self.y_true == y_pred).sum() / self.n_samples

        metrics = dict(
            {
                "target_loss": self.target_loss / self.n_samples,
                "regularization_loss": self.regularization_loss / self.n_samples,
                "total_loss": self.total_loss / self.n_samples,
                "avg_weighted_group_probs": self.avg_weighted_group_probs
                / self.n_samples,
                "y_accuracy": target_acc,
            }
        )
        if self.pixel_probs != 0:
            metrics = metrics | {"pixel_probs": self.pixel_probs / self.n_samples}

        if self.cov_loss != 0:
            metrics = metrics | {"covariance_loss": self.cov_loss / self.n_samples}

        if validation:
            y_metrics = calc_target_metrics(
                self.y_true.numpy(), y_pred_probs.numpy(), config.data
            )

            metrics = metrics | {f"y_{k}": v for k, v in y_metrics.items()}

        return metrics


def create_optimizer(config_model, model):
    """
    Parse the configuration file and return a relevant optimizer object
    """
    assert config_model.optimizer in [
        "sgd",
        "adam",
        "adamw",
    ], "Only SGD and Adam-variant optimizers are available!"

    optim_params = [
        {
            "params": filter(
                lambda p: p.requires_grad, model.parameters()
            ),  # NOTE: Parameters frozen at initialization will never be updated this way
            "lr": config_model.learning_rate,
            "weight_decay": config_model.weight_decay,
        }
    ]

    if config_model.optimizer == "sgd":
        return torch.optim.SGD(optim_params)
    elif config_model.optimizer == "adam":
        return torch.optim.Adam(optim_params)
    elif config_model.optimizer == "adamw":
        return torch.optim.AdamW(optim_params)
