"""
Utility methods for constructing loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional


def create_loss(config):
    """
    Parse configuration file and return a relevant loss function
    """
    if config.model.model == "P2P":
        return P2P_Loss(
            num_classes=config.data.num_classes,
            reduction="mean",
            config=config.model,
        )
    else:
        raise NotImplementedError()


class P2P_Loss(nn.Module):
    """
    Loss function for the concept bottleneck model
    """

    def __init__(
        self,
        num_classes: Optional[int] = 2,
        reduction: str = "mean",
        config: dict = {},
    ) -> None:
        """
        Initializes the loss object

        @param num_classes: the number of the classes of the target variable
        @param reduction: reduction to apply to the output of the CE loss
        @param alpha: parameter controlling the trade-off between the target and concept prediction during the joint
                                        optimization. The higher the @alpha, the higher the weight of the concept prediction loss
        """
        super(P2P_Loss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.use_cov = config.get("use_cov", True)
        self.cov_weight = config.get("cov_weight", 0.0)
        self.reg_variant = config.get("reg_variant", None)
        self.reg_weight_final = config.get("reg_weight", 0.0)
        self.reg_threshold = torch.tensor(config.get("reg_threshold", False))
        self.use_dynamic_threshold = config.get("use_dynamic_threshold", False)
        self.reg_anneal = config.get("reg_anneal", False)
        if self.reg_anneal:
            self.num_epochs = config.num_epochs
            self.compute_reg_weight(0)
        else:
            self.reg_weight = self.reg_weight_final

    def compute_reg_weight(self, epoch):
        if not self.reg_anneal:
            return

        start_epoch = self.num_epochs // 10
        final_epoch = self.num_epochs // 2
        start_value = (
            self.reg_weight_final / 100
        )  # Not 0 s.t. covariance doesn't grow to infinity in the beginning
        if epoch == 0 or epoch >= final_epoch:
            self.reg_weight = self.reg_weight_final
        elif epoch < start_epoch:
            self.reg_weight = start_value
        else:
            self.reg_weight = start_value + (
                (self.reg_weight_final - start_value)
                * (epoch - start_epoch)
                / float(final_epoch - start_epoch)
            )

    def forward(
        self,
        target_pred_logits: Tensor,
        target_true: Tensor,
        groups: Tensor,
        groups_probs: Tensor,
        groups_cov: Tensor,
        model=None,
    ) -> Tensor:
        """
        Computes the total loss for the model, including target prediction loss,
        regularization loss, and covariance loss.

        Args:
            target_pred_logits (Tensor): Predicted logits for the target variable.
            target_true (Tensor): Ground-truth labels for the target variable.
            groups (Tensor): Tensor representing the group assignments for each pixel.
            groups_probs (Tensor): Selection probabilities for each group.
            groups_cov (Tensor): Covariance matrix for the groups.
            model (Optional): The model, used for accessing dynamic thresholds
                            or other model-specific parameters.

        Returns:
            Tensor: A tuple containing:
                - target_loss (Tensor): Cross-entropy loss for the target predictions.
                - reg_loss (Tensor): Regularization loss for group masking.
                - cov_loss (Tensor): Covariance loss for group regularization.
                - total_loss (Tensor): The total loss, which is the sum of target_loss,
                                    reg_loss, and cov_loss.
                - avg_weighted_group_probs (Tensor): Average group selection probability weighted by size of the groups.
        """
        eps = 1e-6
        if self.num_classes == 2:
            # Safety check that everything is still coded with 2 dimensions in binary case
            assert target_pred_logits.size(1) == 2

        target_loss = F.cross_entropy(
            target_pred_logits, target_true, reduction=self.reduction
        )

        if self.use_cov:
            # Regularize Covariance
            cov_loss = self.cov_weight * groups_cov.abs().mean()
        else:
            cov_loss = torch.zeros_like(target_loss)

        # Regularize Group Masking
        # Compute average group selection probabilities, weighted by group size
        avg_weighted_group_probs, group_weights = self.compute_weighted_group_probs(
            model, groups, groups_probs
        )

        # Regularization loss with threshold to not go below it in expectation.
        if self.reg_variant == "l1":
            if self.use_dynamic_threshold:
                self.reg_threshold = model.selector.mask_threshold.squeeze()
            l1_mask = torch.mean(
                torch.max(
                    avg_weighted_group_probs - self.reg_threshold,
                    torch.tensor(0.0),
                )
            )
            reg_loss = self.reg_weight * l1_mask

        # For KL loss, we subtract the threshold from the probabilities s.t.
        # if p -> 1, the loss goes to infinity.
        # This helps annealing by avoiding floating point issues with p~1 for backprop.
        elif self.reg_variant == "kl":
            if self.use_dynamic_threshold:
                self.reg_threshold = model.selector.mask_threshold.squeeze()
                kl_irreducible = -torch.log(1 - self.reg_threshold + eps).mean()
            else:
                kl_irreducible = -torch.log(
                    1 - self.reg_threshold.to(groups.device) + eps
                )
            # If average weight above threshold, regularize average weight
            kl_mask = torch.max(
                avg_weighted_group_probs - self.reg_threshold,
                torch.tensor(0.0),
            )

            reg_loss = (
                self.reg_weight
                * (-torch.log((1 - self.reg_threshold + eps) - kl_mask)).mean()
            ) - self.reg_weight * kl_irreducible

        else:
            raise NotImplementedError()

        total_loss = target_loss + reg_loss + cov_loss
        return target_loss, reg_loss, cov_loss, total_loss, avg_weighted_group_probs

    def compute_weighted_group_probs(self, model, groups, groups_probs):
        """
        Compute the weighted group probabilities
        """
        # Weighting the group probabilities by size of group
        group_weights = model.selector.get_group_average(
            groups, torch.ones_like(groups), reduction="sum"
        )

        # We regularize group probabilities insteda of each pixel separately to avoid shortcuts.
        # I.e. if we regularized predicted pixel probabilities directly, one logit would be huge, while others would be small,
        # leading to loss being small, while group selection probabilities are high, due to influence of the huge logit in the average in logit space for superpixel logit.
        # -> Would lead to high prob for superpixels but low prob for (nearly) all pixels
        # Note: sum_{pixels}[p(pixel)]/n_pixel=sum_{groups}[p(groups)*n_pixel_in_group]/n_pixel
        weighted_group_probs = (
            torch.sum(groups_probs * group_weights, dim=1) / groups[0].numel()
        )
        return weighted_group_probs, group_weights
