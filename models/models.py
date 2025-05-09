"""
Models
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import RelaxedBernoulli, MultivariateNormal
from utils.data import get_normalization_transform
from utils.models import create_classifier, create_importance_predictor
from utils.general import numerical_stability_check


class P2P(nn.Module):
    """
    P2P: Instance-wise Grouped Feature Selection Model
    """

    def __init__(self, config):
        super(P2P, self).__init__()

        # Architectures
        self.selector = Selector(config)
        self.classifier = create_classifier(config)
        self.num_classes = config.data.num_classes
        self.num_monte_carlo = config.model.num_monte_carlo
        # Get value of 0 (for mask) after normalization
        self.min_vals = torch.nn.Parameter(
            get_normalization_transform(config.data.dataset)(
                torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1)
            ).unsqueeze(-1),
            requires_grad=False,
        )

    def forward(self, x, groups, validation=False):
        """
        Forward pass:   1. Selector selects groups of pixels
                        2. Classifier predicts the target variable based on the selected groups

        :param x: input tensor
        :param groups: superpixel group assignments of the pixels
        :return:
            y_pred_logits: predicted logits for the target variable
            groups_probs: selection probabilities of the superpixels
            pixel_probs: selection probabilities for each separate pixel
            classifier_input: masked input to the classifier
            groups_cov: covariance matrix of the groups
        """
        # During validation, we need to compute the optimal dynamic masking threshold
        # and store it in self.mask_threshold which is then used in the selector
        if validation and self.selector.use_dynamic_threshold:
            self.selector.mask_threshold = self.find_optimal_threshold(x, groups)

        mask, groups_probs, pixel_probs, groups_cov = self.selector(
            x, groups, validation
        )
        # Create masked input image for classifier
        classifier_input = x.unsqueeze(-1).expand(
            -1, -1, -1, -1, self.num_monte_carlo
        ) * mask + self.min_vals * (
            1 - mask
        )  # = self.min_vals + (x-self.min_vals) * mask
        y_pred_logits = self.predict_from_input(classifier_input)

        if validation:
            return (
                y_pred_logits,
                groups_probs,
                pixel_probs,
                classifier_input,
                groups_cov,
            )

        return y_pred_logits, groups_probs, pixel_probs, groups_cov

    def predict_from_input(self, classifier_input):
        """Predict the target variable from the masked input image"""
        y_pred_probs = torch.zeros(
            classifier_input.size(0),
            self.num_classes,
            device=classifier_input.device,
        )
        # We could put mcmc dimension into batch dimension for parallelization, but this would require a lot of memory
        for i in range(self.num_monte_carlo):
            y_pred = self.classifier(classifier_input[..., i])
            y_pred_probs += F.softmax(y_pred, dim=1)
        y_pred_probs /= self.num_monte_carlo
        y_pred_logits = torch.log(y_pred_probs + 1e-6)
        return y_pred_logits

    def find_optimal_threshold(self, x, groups):
        """
        Used at validation time to find the optimal dynamic masking threshold.
        Find the optimal dynamic masking threshold for each sample such
        that the classifier's certainty is above the defined certainty threshold.
        """
        # Get value at intermediate hooks to input different masking thresholds
        _ = self.selector.importance_predictor(x)
        low_hook_input = self.selector.importance_predictor.model.conv_input["low"]
        high_hook_input = self.selector.importance_predictor.model.conv_input["high"]
        # Initialize the threshold to the strongest masking
        mask_threshold = (
            torch.ones(x.shape[0], device=x.device) * self.selector.min_threshold
        ).view(-1, 1, 1, 1)
        above_threshold = torch.zeros(x.shape[0], device=x.device).bool()
        # For each sample, we decrease the masking strength (i.e. increase the masking threshold)
        # until classifier is certain to make a prediction
        while mask_threshold.max() < 1.0 and not above_threshold.all():
            # Giving mask threshold as input to the selector by adding it to intermediate representation
            low_input = low_hook_input + mask_threshold
            high_input = high_hook_input + mask_threshold
            # Make group selection given the masking threshold
            importance_emb = self.selector.importance_predictor.threshold_as_input(
                low_input, high_input, x.shape[-2:]
            )
            groups_mcmc, _, _, _ = self.selector.select_groups(
                importance_emb, groups, validation=True
            )
            mask = self.selector.groups_to_mask(
                groups, groups_mcmc, x.shape + (self.num_monte_carlo,)
            )
            # Create masked input image for classifier
            classifier_input = x.unsqueeze(-1).expand(
                -1, -1, -1, -1, self.num_monte_carlo
            ) * mask + self.min_vals * (
                1 - mask
            )  # = self.min_vals + (x-self.min_vals) * mask
            # Make prediction
            y_pred_logits = self.predict_from_input(classifier_input)
            # Get certainty of prediction
            y_pred_probs = F.softmax(y_pred_logits, dim=-1)
            # Check for which samples the classifier has a certainty above the threshold
            # Once it's certain, it can't be reversed (otherwise we might run into multiple-testing issues due to sampling randomness)
            above_threshold = above_threshold | (
                y_pred_probs.max(1)[0] >= self.selector.certainty_threshold
            )
            # Add more groups by increasing masking threshold to the images for which classifier is still uncertain.
            mask_threshold += (1 - above_threshold.float()).view(
                -1, 1, 1, 1
            ) * self.selector.min_threshold
        mask_threshold = mask_threshold.clamp(max=1.0)
        return mask_threshold


class Selector(nn.Module):
    """
    The Selector module is responsible for selecting important groups of pixels.
    Args:
        config (DictConfig): Configuration object containing model and data parameters.

    Attributes:
        importance_predictor (nn.Module): A neural network that predicts importance embeddings for each pixels.
        activation (nn.Module): Sigmoid activation function for computing probabilities.
        straight_through (bool): Whether to use the straight-through Gumbel-Softmax relaxation.
        final_temp (float): Final temperature for Gumbel-Softmax annealing.
        num_monte_carlo (int): Number of Monte Carlo samples for probabilistic group selection.
        final_epoch (int): Epoch at which annealing ends.
        n_segments (int): Number of groups (i.e. superpixels).
        use_cov (bool): Whether to use group covariance.
        use_dynamic_threshold (bool): Whether to use dynamic thresholds for masking.
        min_threshold (float): Lowest possible threshold for dynamic masking.
        max_threshold (float): Highest possible threshold for dynamic masking.
        certainty_threshold (float): Certainty threshold for dynamic masking.
    """

    def __init__(self, config):
        super(Selector, self).__init__()

        self.importance_predictor = create_importance_predictor(config)
        # self.group_proposer = create_proposer(config)
        self.activation = nn.Sigmoid()
        self.straight_through = config.model.straight_through
        self.final_temp = config.model.final_temp
        self.num_monte_carlo = config.model.num_monte_carlo
        self.final_epoch = config.model.num_epochs // 2
        self.n_segments = config.data.n_segments
        self.use_cov = config.model.use_cov
        self.use_dynamic_threshold = config.model.use_dynamic_threshold
        if self.use_dynamic_threshold:
            self.min_threshold = 0.05
            self.max_threshold = 1.0
            if isinstance(config.model.certainty_threshold, float):
                self.certainty_threshold = config.model.certainty_threshold
            else:
                # Setting a value during validation, will be iterated over at test
                self.certainty_threshold = config.model.certainty_threshold[0]
        self.compute_temperature(0, init=True)

    def forward(self, x, groups, validation=False):
        """
        Forward pass of the Selector module. Computes importance embeddings per pixel,
        aggregates them to group level selection probabilities,
        and samples groups based on their probabilities.

        Args:
            x (Tensor): Input images.
            groups (Tensor): Tensor representing superpixel group assignments for each pixel.
            validation (bool, optional): Whether the forward pass is for validation. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - mask (Tensor): Binary mask indicating selected pixels.
                - groups_probs (Tensor): Selection probabilities for each group.
                - pixel_probs (Tensor): Selection probabilities for each pixel.
                - groups_cov (Tensor): Covariance matrix for the groups (if use_cov is True).
        """
        importance_emb = self.get_importance_emb(x, validation=validation)
        groups_mcmc, groups_probs, pixel_probs, groups_cov = self.select_groups(
            importance_emb, groups, validation
        )
        mask = self.groups_to_mask(
            groups, groups_mcmc, x.shape + (self.num_monte_carlo,)
        )

        return mask, groups_probs, pixel_probs, groups_cov

    def get_importance_emb(self, x, validation=False):
        """
        Computes the importance embeddings for the input tensor. If dynamic thresholds are enabled,
        we provide the model's internal representation with information about the thresholds.

        Args:
            x (Tensor): Input
            validation (bool, optional): Whether the embeddings are being computed for validation.

        Returns:
            Tensor: Importance embeddings for the input tensor.
        """
        importance_emb = self.importance_predictor(x)
        if not self.use_dynamic_threshold:
            return importance_emb
        else:
            # At training time, randomly sample the threshold
            if not validation:
                self.mask_threshold = (
                    torch.empty(x.shape[0], device=x.device)
                    .uniform_(self.min_threshold, self.max_threshold)
                    .view(-1, 1, 1, 1)
                )
            # At validation time, we have previously computes the optimal threshold
            # with find_optimal_threshold and stored it in self.mask_threshold
            low_input = (
                self.importance_predictor.model.conv_input["low"] + self.mask_threshold
            )
            high_input = (
                self.importance_predictor.model.conv_input["high"] + self.mask_threshold
            )
            # Giving mask threshold as input to the selector by adding it to intermediate representation
            # and continue forward pass after the hook
            importance_emb = self.importance_predictor.threshold_as_input(
                low_input, high_input, x.shape[-2:]
            )
            return importance_emb

    def select_groups(self, importance_emb, groups, validation=False):
        """
        Selects groups based on their importance embeddings.

        Args:
            importance_emb (Tensor): Pixelwise importance embeddings for the input.
            groups (Tensor): Superpixel group assignments for each pixel.
            validation (bool, optional): Whether the group selection is for validation. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - groups_mcmc (Tensor): Monte Carlo samples of selected groups.
                - group_probs (Tensor): Selection probabilities for each group.
                - pixel_probs (Tensor): Selection probabilities for each pixel.
                - groups_cov (Tensor): Covariance matrix for the groups (if use_cov is True).
        """
        # For each group, average the embeddings of the pixels assigned to that group
        # Such that now we have an embedding per-group
        groups_avg = self.get_group_average(groups, importance_emb)

        # Compute groupwise log-normal distribution
        # Probabilities are the first slice, embedding in the rest
        groups_mu = groups_avg[:, 0]
        if self.use_cov:
            groups_emb = groups_avg[:, 1:]

            # Compute covariance matrix via dot product of embeddings
            groups_sigma = groups_emb.mT @ groups_emb
        # For logging pixelwise probabilities:
        logits = importance_emb[:, 0]

        # Compute sigmoid(logits), sigmoid(groups_mu) for probabilities per-pixel, per-group for regularization
        pixel_probs = self.activation(logits)
        group_probs = self.activation(groups_mu)

        if self.use_cov:
            groups_sigma = numerical_stability_check(groups_sigma)
            if validation:
                # No sampling at validation time but taking mean instead
                groups_mcmc_logits = groups_mu.unsqueeze(-1).expand(
                    -1, -1, self.num_monte_carlo
                )
                groups_mcmc_probs = self.activation(groups_mcmc_logits)
            else:
                # Sample probabilities from log-normal distribution
                dist_lognorm = MultivariateNormal(groups_mu, groups_sigma)
                groups_mcmc_logits = dist_lognorm.rsample(
                    [self.num_monte_carlo]
                ).movedim(0, -1)
                groups_mcmc_probs = self.activation(groups_mcmc_logits)
        else:
            groups_sigma = None
            groups_mcmc_logits = groups_mu.unsqueeze(-1).expand(
                -1, -1, self.num_monte_carlo
            )
            groups_mcmc_probs = self.activation(groups_mcmc_logits)

        # Sample selected groups
        if validation:
            # We use hard thresholding at validation time
            groups_mcmc = (groups_mcmc_probs > 0.5) * 1.0

            return groups_mcmc, group_probs, pixel_probs, groups_sigma
        else:
            # Backpropagation necessary: Gumbel-Softmax Bernoulli relaxation
            dist_gumbel = RelaxedBernoulli(
                temperature=self.curr_temp, probs=groups_mcmc_probs
            )
            groups_mcmc_relaxed = dist_gumbel.rsample()
            if self.straight_through:
                # Straight-Through Gumbel Softmax
                groups_mcmc_hard = (groups_mcmc_relaxed > 0.5) * 1
                groups_mcmc = (
                    groups_mcmc_hard
                    - groups_mcmc_relaxed.detach()
                    + groups_mcmc_relaxed
                )
            else:
                groups_mcmc = groups_mcmc_relaxed

        return groups_mcmc, group_probs, pixel_probs, groups_sigma

    def groups_to_mask(self, groups, groups_mcmc, shape):
        """
        Mapper for selected groups to corresponding selected pixels as binary mask.

        Args:
            groups (Tensor): Tensor representing superpixel group assignments for each pixel.
            groups_mcmc (Tensor): Monte Carlo samples of selected groups.
            shape (Tuple[int]): Shape of the input tensor, including the Monte Carlo dimension.

        Returns:
            Tensor: Binary mask.
        """
        assert len(shape) == 5
        # For each pixel, look up whether its assigned group(->index) was selected(->input) and create a mask
        mask = torch.gather(
            dim=1,
            index=groups.unsqueeze(-1).expand(-1, -1, -1, shape[-1]).flatten(1, 2),
            input=groups_mcmc,
        )
        # Expand by RGB channels
        mask = (
            torch.unflatten(mask, 1, shape[2:4])
            .unsqueeze(1)
            .expand(-1, shape[1], -1, -1, -1)
        )

        return mask

    def get_group_average(self, groups, embedding, reduction="mean"):
        """
        Aggregate features of pixels within a group, for each group.

        Args:
            groups (Tensor): Superpixel group assignments for each pixel.
            embedding (Tensor): Feature (e.g. embedding) to be aggregated across pixels of same group.
            reduction (str, optional): Reduction method to apply ('mean' or 'sum').

        Returns:
            Tensor: Group-aggregated features.
        """

        if embedding.ndim == 3:
            squeeze_at_end = True
            embedding = embedding.unsqueeze(1)
        else:
            squeeze_at_end = False

        n_segments = max(groups.max() + 1, self.n_segments)
        groups_emb = torch.zeros(
            (embedding.shape[0], embedding.shape[1], n_segments),
            device=embedding.device,
            dtype=embedding.dtype,
        )
        # Flatten the group tensor s.t. scatter can iterate through a single dimension. I.e. 2D grid becomes 1D
        # Then, expand/repeat the group tensor by embedding dimensions as scatter requires elementwise operations --> tensor now has (identical) group assignments for each dimension at a given location
        flattened_groups = (
            groups.flatten(1, -1).unsqueeze(1).expand(-1, embedding.shape[1], -1)
        )
        # Flatten the embedding tensor s.t. scatter can iterate through a single dimension. I.e. 2D grid becomes 1D
        flattened_embedding = embedding.flatten(2, -1)

        # For each value=flattened_embedding[i,j,k], look up the group_index=groups[i,j,k] and add the value to the corresponding group_dims[i,j,group_index].
        # By some magic, we can average the values for each group instead of only summing
        groups_emb.scatter_reduce_(
            dim=2,
            index=flattened_groups,
            src=flattened_embedding,
            reduce=reduction,
            include_self=False,
        )
        if squeeze_at_end:
            groups_emb = groups_emb.squeeze(1)

        return groups_emb

    def get_embedding(self, x, groups, validation=False):
        """
        Computes the pixel and group embeddings for the input tensor.

        Args:
            x (Tensor): Input image.
            groups (Tensor): Superpixel group assignments for each pixel.
            validation (bool, optional): Whether the embeddings are being computed for validation. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor]:
                - pixel_emb (Tensor): Embeddings for each pixel.
                - groups_emb (Tensor): Embeddings for each group.
        """
        importance_emb = self.get_importance_emb(x, validation=validation)
        # For each group, average the embeddings of the pixels assigned to that group
        groups_avg = self.get_group_average(groups, importance_emb)

        # Probabilities are the first slice, embedding in the rest
        pixel_emb = importance_emb[:, 1:]
        groups_emb = groups_avg[:, 1:]
        return pixel_emb, groups_emb

    def compute_temperature(self, epoch, init=False):
        """
        Computes the temperature for the Gumbel-Softmax relaxation based on the current epoch.
        The temperature is annealed from an initial value to a final value over a specified number of epochs.
        """
        final_temp = torch.tensor([self.final_temp])
        init_temp = torch.tensor([1.0])
        if self.final_epoch > 0:
            rate = (math.log(final_temp) - math.log(init_temp)) / float(
                self.final_epoch
            )
            curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        else:
            curr_temp = final_temp
        if init:
            self.curr_temp = torch.nn.Parameter(curr_temp, requires_grad=False)
        else:
            self.curr_temp.copy_(curr_temp)
