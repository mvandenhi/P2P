import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from torchmetrics import AveragePrecision
from skimage.segmentation import mark_boundaries
import wandb
from utils.data import get_normalization_transform
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import io
import cv2
from tqdm import tqdm
from torch import nn
import matplotlib.lines as mlines
from utils.data import get_id_to_label


def compute_fidelity(
    loader,
    model,
    config,
    device,
    thresholds=None,
    inverse=False,
):
    """
    Evaluate the model's insertion and delection fidelity at different thresholds.
    """

    model.eval()
    if thresholds is None:
        thresholds = [
            0.0,
            0.01,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.7,
            0.8,
            0.9,
            0.99,
            1.0,
        ]
    with torch.no_grad():

        y_pred_logits_data = torch.empty(
            [len(loader.dataset), config.data.num_classes, len(thresholds)]
        )
        # NOTE that y_target is the class that is predicted without modifications. it is NOT the true label
        y_target = torch.empty(
            [len(loader.dataset), len(thresholds)],
            dtype=torch.long,
        )

        for k, batch in enumerate(tqdm(loader, position=0, leave=True)):
            x, groups, _ = (
                batch[0][0].to(device),
                batch[0][1].to(device),
                batch[1].to(device),
            )

            (
                target_pred_logits,
                groups_probs,
                _,
                masked_x,
                _,
            ) = model(x, groups, validation=True)

            # Mask group with highest selection probability.
            # Multiply it with it's associated weight=size to measure how much of image it covers.
            # Continue until we're below x% of the total probability.
            group_weights = (
                model.selector.get_group_average(
                    groups, torch.ones_like(groups), reduction="sum"
                )
                / groups[0].numel()
            )

            # Order groups by their selection probability and get their size
            groups_indices = torch.argsort(groups_probs, descending=True, dim=1)
            group_weights_ordered = torch.gather(group_weights, 1, groups_indices)

            # Note that since November 2024, cumsum is deterministic, however, for backward compatibility, we retain the non-deterministic behavior
            # Cumsum to know when we are below x% of the selection probability
            torch.use_deterministic_algorithms(False)
            groups_probs_cumsum = torch.cumsum(group_weights_ordered, 1)
            torch.use_deterministic_algorithms(True)

            for idx, threshold in enumerate(thresholds):
                # Create a boolean mask where the cumulative sum exceeds the threshold
                cumsum_mask = groups_probs_cumsum.ge(threshold)

                if inverse:
                    # Mask all groups from least important to most imporant until over threshold
                    # For insertion fidelity
                    cumsum_mask = ~cumsum_mask
                else:
                    # Mask all groups from most important to least imporant until over threshold
                    # Also mask first group after threshold
                    cumsum_mask.scatter_(
                        1, torch.max(cumsum_mask, 1).indices.unsqueeze(1), False
                    )

                # Backtransform mask to original order
                group_mask = torch.scatter(
                    groups_probs, 1, groups_indices, cumsum_mask.float()
                )

                # Mask pixels assiciated with the masked groups
                mask = model.selector.groups_to_mask(
                    groups,
                    group_mask.unsqueeze(-1).expand(-1, -1, model.num_monte_carlo),
                    x.shape + (model.num_monte_carlo,),
                )
                classifier_input = masked_x * mask + model.min_vals * (1 - mask)

                # Get the prediction from the classifier
                y_pred_logits = model.predict_from_input(classifier_input)

                # Store the prediction and the true label
                y_pred_logits_data[
                    k * loader.batch_size : k * loader.batch_size
                    + y_pred_logits.shape[0],
                    :,
                    idx,
                ] = y_pred_logits.cpu()
                y_target[
                    k * loader.batch_size : k * loader.batch_size
                    + y_pred_logits.shape[0],
                    idx,
                ] = target_pred_logits.argmax(1).cpu()

    # Compute performance with respect to initial prediction
    y_pred_probs = nn.Softmax(dim=1)(y_pred_logits_data)
    y_pred = y_pred_logits_data.argmax(dim=1)
    target_acc = (y_target == y_pred).sum(0) / len(y_target)

    metrics = {}
    if inverse:
        thresholds = [1 - t for t in thresholds]

    for i in range(len(thresholds)):
        if thresholds[i] == 0.0:
            # Fixing this, as with no masking, it's same as before, but because we coded to mask "first after threshold", we need to fix this here.
            target_acc[i] = 1.0
        if (not inverse and thresholds[i] not in [0.05, 0.1, 0.2]) or (
            inverse and thresholds[i] not in [0.8, 0.9, 0.95]
        ):
            continue
        y_metrics = calc_target_metrics(
            y_target[:, i].numpy(), y_pred_probs[:, :, i].numpy(), config.data
        )
        metrics[f"Accuracy@{thresholds[i]:.2f}_masked"] = target_acc[i].item()
        metrics.update(
            {f"{k}@{thresholds[i]:.2f}_masked": v for k, v in y_metrics.items()}
        )

    if inverse:
        # Invert the thresholds for plotting s.t. for insertion 0% means black image
        data = [[1 - threshold, acc] for threshold, acc in zip(thresholds, target_acc)][
            ::-1
        ]
    else:
        data = [[threshold, acc] for threshold, acc in zip(thresholds, target_acc)]

    table = wandb.Table(data=data, columns=["Threshold", "Accuracy"])
    if not inverse:
        print("LOWER IS BETTER")
        wandb_name = "deletion_fidelity"
        wandb_curve_name = "deletion_fidelity_curve"
        curve_title = "Deletion Fidelity Curve"
        prints = f"Deletion Fidelity: "

    else:
        print("HIGHER IS BETTER")
        wandb_name = "insertion_fidelity"
        wandb_curve_name = "insertion_fidelity_curve"
        curve_title = "Insertion Fidelity Curve"
        prints = f"Insertion Fidelity: "

    if config.model.use_dynamic_threshold:
        wandb_name += f"_threshold_{model.selector.certainty_threshold}"
        wandb_curve_name += f"_threshold_{model.selector.certainty_threshold}"
        curve_title += f" Certainty Threshold: {model.selector.certainty_threshold} "
        prints += f" Certainty Threshold: {model.selector.certainty_threshold} "

    wandb.log({f"{wandb_name}/{k}": v for k, v in metrics.items()})
    # Log the fidelity curve using wandb.plot.line
    wandb.log(
        {
            wandb_curve_name: wandb.plot.line(
                table, "Threshold", "Accuracy", title=curve_title
            )
        }
    )
    for key, value in sorted(metrics.items(), reverse=inverse):
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    # Store wandb curve with target_acc as y, and thresholds as x

    return None


def compute_localization(loader, model, device, config):
    """
    Evaluate the model's localization. We use mask_perc_corr in the paper because
    maskIoU is optimal if the full object is selected, while we want to minimize the
    number of selected pixels within the object.
    """
    model.eval()
    evaluator = MaskEvaluator()
    maskiou = 0.0
    mask_perc_corr = 0.0
    n_samples = 0
    with torch.no_grad():
        for k, batch in enumerate(tqdm(loader, position=0, leave=True)):
            x, groups, gt_mask = (
                batch[0][0].to(device),
                batch[0][1].to(device),
                batch[2].to(device),
            )

            (
                _,
                groups_probs,
                _,
                _,
                _,
            ) = model(x, groups, validation=True)

            # Get probabilistic mask required for PxAP
            mask = model.selector.groups_to_mask(
                groups,
                groups_probs.unsqueeze(-1).expand(-1, -1, 1),
                x.shape[0:1] + (1,) + x.shape[2:4] + (1,),
            )
            explanation = mask.squeeze()

            # Get mask for maskIoU
            # Use the group probabilities to select groups with hard threshold
            groups_hard = (groups_probs > 0.5) * 1.0
            mask_hard = model.selector.groups_to_mask(
                groups,
                groups_hard.unsqueeze(-1).expand(-1, -1, 1),
                x.shape[0:1] + (1,) + x.shape[2:4] + (1,),
            )
            evaluator.accumulate(explanation, gt_mask)
            # Compute maskIoU
            maskiou += compute_maskiou(
                mask_hard.squeeze().to(torch.bool), gt_mask
            ).sum()
            mask_perc_corr += compute_pred_correctness(
                mask_hard.squeeze().to(torch.bool), gt_mask
            ).sum()
            n_samples += mask_hard.shape[0]
    if not config.model.use_dynamic_threshold:
        pxap_name = "localization/pxap"
        maskiou_name = "localization/maskiou"
        mask_perc_corr_name = "localization/mask_perc_corr"
    else:
        thresh = model.selector.certainty_threshold
        print("Certainty Threshold: ", thresh)
        pxap_name = f"localization_threshold_{thresh}/pxap"
        maskiou_name = f"localization_threshold_{thresh}/maskiou"
        mask_perc_corr_name = f"localization_threshold_{thresh}/mask_perc_corr"

    pxap = evaluator.compute()
    print("HIGHER IS BETTER")
    wandb.log({pxap_name: pxap})
    print(f"Pixel Average Precision (PxAP): {pxap:.3f}")

    maskiou /= n_samples
    print("HIGHER IS BETTER")
    wandb.log({maskiou_name: maskiou})
    print(f"Mask IoU: {maskiou:.3f}")

    mask_perc_corr /= n_samples
    wandb.log({mask_perc_corr_name: mask_perc_corr})
    print(f"Mask Percentage Correct: {mask_perc_corr:.3f}")
    return None


def calc_target_metrics(ys, scores_pred, config, n_decimals=4, n_bins_cal=10):
    """Computing AUROC and AUPR"""
    # AUROC
    if config.num_classes == 2:
        auroc = roc_auc_score(ys, scores_pred)
    elif config.num_classes > 2:
        auroc = _roc_auc_score_with_missing(ys, scores_pred)

    # AUPR
    aupr = 0.0
    if config.num_classes == 2:
        aupr = average_precision_score(ys, scores_pred)
    elif config.num_classes > 2:
        ap = AveragePrecision(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        aupr = float(
            ap(torch.tensor(scores_pred), torch.tensor(ys.squeeze()).type(torch.int64))
            .cpu()
            .numpy()
        )

    return {
        "AUROC": np.round(auroc, n_decimals),
        "AUPR": np.round(aupr, n_decimals),
    }


def _roc_auc_score_with_missing(labels, scores):
    """Computes OVR macro-averaged AUROC under missing classes"""
    aurocs = np.zeros((scores.shape[1],))
    weights = np.zeros((scores.shape[1],))
    for c in range(scores.shape[1]):
        if len(labels[labels == c]) > 0 and len(np.unique(labels)) > 1:
            labels_tmp = (labels == c) * 1.0
            aurocs[c] = roc_auc_score(
                labels_tmp, scores[:, c], average="weighted", multi_class="ovr"
            )
            weights[c] = len(labels[labels == c])
        else:
            aurocs[c] = np.NaN
            weights[c] = np.NaN

    # Computing weighted average
    mask = ~np.isnan(aurocs)
    weighted_sum = np.sum(aurocs[mask] * weights[mask])
    average = weighted_sum / len(labels)
    # Regular "macro"
    # average = np.nanmean(aurocs)
    return average


def log_images(
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
):
    """Logging visualizations of the model's predictions and explanations."""
    log_id = np.random.choice(x.shape[0])
    ## Log image overlayed with groups
    # Extract the image and segmentation mask
    img = (
        x[log_id].permute(1, 2, 0).cpu().numpy()
    )  # Convert tensor to HWC format for display
    seg = groups[log_id].cpu().numpy()

    # Draw boundaries on the original image
    img_with_boundaries = mark_boundaries(
        img, seg, color=(1, 0, 0), mode="inner"
    )  # Red color for boundaries

    # Log masked image; using the first MCMC sample
    if masked_x.ndim == 4:
        masked_img = masked_x[log_id].permute(1, 2, 0).cpu().numpy()
    else:
        masked_img = masked_x[log_id][..., 0].permute(1, 2, 0).cpu().numpy()
    norm_transform = get_normalization_transform(config.data.dataset)

    # Concatenate the image overlayed with groups and the actual masked image
    combined_img = np.concatenate((img_with_boundaries, masked_img), axis=1)

    combined_img = combined_img * np.array(norm_transform.std) + np.array(
        norm_transform.mean
    )

    combined_img_int = np.uint8(combined_img * 255)

    correct_class = target_true[log_id].item()
    if isinstance(target_pred_logits, tuple):
        predicted_class = target_pred_logits[0].argmax(1)[log_id].item()
    else:
        predicted_class = target_pred_logits.argmax(1)[log_id].item()

    correct = predicted_class == correct_class
    id_to_label = get_id_to_label(config.data)

    # Log the combined image with separate captions
    wandb.log(
        {
            "image_groups": [
                wandb.Image(
                    Image.fromarray(combined_img_int),
                    caption=(
                        f"Image with Group Boundaries (left) and Masked Groups (right), "
                        f"Classified correctly: {correct}, "
                        f"Predicted label: {id_to_label[predicted_class]}, "
                        f"True label: {id_to_label[correct_class]}"
                    ),
                )
            ],
        }
    )

    ## Log masked image including heatmap
    img_rescaled = (
        img * np.array(norm_transform.std) + np.array(norm_transform.mean)
    ).clip(0, 1)

    # Probabilities to heatmap
    heatmap_img_pixel = cv2.applyColorMap(
        np.uint8(pixel_probs[log_id].cpu().squeeze() * 255), cv2.COLORMAP_JET
    )
    # BGR to RGB ordering to combine images
    heatmap_img_pixel = cv2.cvtColor(heatmap_img_pixel, cv2.COLOR_BGR2RGB)
    # Overlay heatmap to original image
    superimposed_img_pixel = cv2.addWeighted(
        heatmap_img_pixel, 0.5, np.uint8(img_rescaled * 255), 0.5, 0
    )

    groups_probs = groups_probs[log_id]
    pixelprobs_by_group = torch.gather(
        input=groups_probs, index=groups[log_id].view(-1), dim=0
    ).view(groups[log_id].shape)
    heatmap_img_group = cv2.applyColorMap(
        np.uint8(pixelprobs_by_group.cpu() * 255), cv2.COLORMAP_JET
    )
    heatmap_img_group = cv2.cvtColor(heatmap_img_group, cv2.COLOR_BGR2RGB)
    superimposed_img_group = cv2.addWeighted(
        heatmap_img_group, 0.5, np.uint8(img_rescaled * 255), 0.5, 0
    )

    superimposed_img = np.concatenate(
        (superimposed_img_pixel, superimposed_img_group), axis=1
    )

    heatmaps_img = np.concatenate((heatmap_img_pixel, heatmap_img_group), axis=1)

    combined_img = np.concatenate(
        (combined_img_int, superimposed_img, heatmaps_img), axis=0
    )
    # Log the combined image with separate captions
    wandb.log(
        {
            "image_groups_combined": [
                wandb.Image(
                    Image.fromarray(combined_img),
                    caption=(
                        f"Top: Image with Group Boundaries (left) and Masked Groups (right), "
                        f"Bottom: Pixel-wise and Group-wise Selection Heatmaps, Classified correctly: {correct}, "
                        f"Predicted label: {id_to_label[predicted_class]}, "
                        f"True label: {id_to_label[correct_class]}"
                    ),
                    file_type="jpg",
                )
            ],
        }
    )
    if not model.selector.use_cov:
        return None
    ## Log groups' dependencies somehow
    # Sample correlation matrix
    corr_matrix = correlation_from_covariance(groups_cov[log_id].cpu().numpy())[
        : seg.max() + 1, : seg.max() + 1
    ]

    # Load all images in advance
    groups_img = groups[log_id].cpu().numpy()

    groups_vis_list = []
    # Get visualization of each superpixel
    for i in range(groups_img.max() + 1):
        masked_image = apply_bounding_box(img_rescaled, groups_img == i)
        groups_vis_list.append(masked_image)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(32, 32))
    sns.heatmap(
        corr_matrix,
        ax=ax,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar=False,
    )

    # Turn off default labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Remove axis ticks
    ax.tick_params(left=False, bottom=False, top=False, direction="out")

    # Adjust the axis limits to make space for images
    ax.set_xlim(-20, None)
    ax.set_ylim(None, -20)  # Reverse the y-axis for proper orientation
    for i, group_vis in enumerate(groups_vis_list):
        imagebox = OffsetImage(group_vis, zoom=1)  # Adjust zoom as needed
        if i % 3 == 0:
            offset = 0.0
        elif i % 3 == 1:
            offset = -4.75
        else:
            offset = -9.5
        ab = AnnotationBbox(
            imagebox,
            (i + 0.5, -0.7 + offset),  # Position just below the x-axis
            frameon=False,
            xycoords="data",
            boxcoords="data",
            pad=0,
            box_alignment=(0.5, 0),
        )
        ax.add_artist(ab)
        xmin = xmax = i + 0.5
        ymin = 0
        ymax = -0.7 + offset
        l = mlines.Line2D([xmin, xmax], [ymin, ymax], color="black", linewidth=0.5)
        ax.add_line(l)

    # Add images as labels on the y-axis
    for i, group_vis in enumerate(groups_vis_list):
        imagebox = OffsetImage(group_vis, zoom=1)  # Adjust zoom as needed
        if i % 3 == 0:
            offset = 0.0
        elif i % 3 == 1:
            offset = -4.75
        else:
            offset = -9.5
        ab = AnnotationBbox(
            imagebox,
            (-0.7 + offset, i + 0.5),  # Position just to the left of the y-axis
            frameon=False,
            xycoords="data",
            boxcoords="data",
            pad=0,
            box_alignment=(1, 0.5),
        )
        ax.add_artist(ab)
        ymin = ymax = i + 0.5
        xmin = -0.7 + offset
        xmax = 0
        l = mlines.Line2D([xmin, xmax], [ymin, ymax], color="black", linewidth=0.5)
        ax.add_line(l)

    plt.tight_layout()

    # Save the heatmap to a file
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close()
    wandb.log(
        {
            "Correlation Matrix": wandb.Image(
                Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=256),
                file_type="png",
            )
        }
    )

    pixel_emb, groups_emb = model.selector.get_embedding(x, groups, validation=True)
    if config.model.get("num_dims", False) == 4:
        # Plot the embedding of the pixels and group
        pixel_emb = pixel_emb[log_id].permute(1, 2, 0).cpu().numpy()
        pixelemb_by_group = torch.gather(
            input=groups_emb[log_id],
            index=groups[log_id].view(-1).unsqueeze(0).repeat((3, 1)),
            dim=1,
        ).view(groups[log_id].unsqueeze(0).repeat((3, 1, 1)).shape)

        pixelemb_by_group = pixelemb_by_group.permute(1, 2, 0).cpu().numpy()

        # Create heatmap from pixel embeddings
        pixel_emb_rescaled = (pixel_emb - pixel_emb.min((0, 1))) / (
            pixel_emb.max((0, 1)) - pixel_emb.min((0, 1))
        )
        heatmap_img_pixel = np.uint8(pixel_emb_rescaled * 255)

        # Create heatmap from group embeddings
        group_emb_rescaled = (pixelemb_by_group - pixelemb_by_group.min((0, 1))) / (
            pixelemb_by_group.max((0, 1)) - pixelemb_by_group.min((0, 1))
        )
        heatmap_img_group = np.uint8(group_emb_rescaled * 255)

        # Superimpose heatmap on original image
        superimposed_img_pixel = cv2.addWeighted(
            heatmap_img_pixel, 0.5, np.uint8(img_rescaled * 255), 0.5, 0
        )
        superimposed_img_group = cv2.addWeighted(
            heatmap_img_group, 0.5, np.uint8(img_rescaled * 255), 0.5, 0
        )

        # Create 3x2 grid of images, where mid are the superimposed images and bottom are the heatmaps only
        combined_img = np.concatenate(
            (superimposed_img_pixel, superimposed_img_group), axis=1
        )
        heatmaps_img = np.concatenate((heatmap_img_pixel, heatmap_img_group), axis=1)
        combined_img = np.concatenate((combined_img, heatmaps_img), axis=0)
        combined_img = np.concatenate((combined_img_int, combined_img), axis=0)

        # Caption: Compute whether groupwise-embeddings describe two groups:
        # The one with the highest selection probability and the one with the lowest selection probability
        # 1. Compute 2-means clustering on the group embeddings and the variability to mean
        k_means = KMeans(n_clusters=2, random_state=0, n_init="auto")
        cluster_ass = k_means.fit_predict(pixel_emb_rescaled.reshape(-1, 3))
        dist = k_means.transform(pixel_emb_rescaled.reshape(-1, 3)).min(1)
        avg_cluster_dist = dist.mean()

        # 2. Compute overlap between embedding clustering and selection probability
        ## Recover mask
        mask = (
            np.uint(
                (
                    masked_img * np.array(norm_transform.std)
                    + np.array(norm_transform.mean)
                )
                * 255
            )
            > 0
        ).all(-1)
        # Compute Overlap
        overlap = (mask.reshape(-1) == cluster_ass).mean()
        overlap = max(overlap, 1 - overlap)  # Switching arbitrary order of 0/1

        # Log the combined image with separate captions
        wandb.log(
            {
                "image_embedding_combined": [
                    wandb.Image(
                        Image.fromarray(combined_img),
                        caption=(
                            f"Top: Image with Group Boundaries (left) and Masked Groups (right), "
                            f"Middle: Image with Pixelwise Embedding (left) and Groupwise Embedding (right), "
                            f"Bottom: Pixel-wise and Group-wise Selection Heatmaps, "
                            f"Variability of embedding wrt 2 clusters: {avg_cluster_dist:.4f}, "
                            f"Overlap of 2-means clustering with selection: {overlap:.4f}"
                        ),
                        file_type="jpg",
                    )
                ],
            }
        )

        # Log covariance as color but only the activated parts
        mask_expanded = np.tile(np.expand_dims(mask, axis=-1), (3, 1, 1))
        new_combined_img = combined_img[:, combined_img.shape[1] // 2 :] * mask_expanded
        # Log the combined image with separate captions
        wandb.log(
            {
                "image_embedding_combined_masked": [
                    wandb.Image(
                        Image.fromarray(new_combined_img),
                        caption=(
                            f"Top: Image with Masked Groups, "
                            f"Middle: Image with Groupwise Embedding, "
                            f"Bottom: Groupwise Embedding, "
                        ),
                        file_type="jpg",
                    )
                ],
            }
        )

    # Get active groups
    img_rescaled = np.uint8(
        (masked_img * np.array(norm_transform.std) + np.array(norm_transform.mean))
        * 255
    )

    idx = []
    for i in range(groups[log_id].max()):
        group_idx = torch.where(groups[log_id].reshape(-1) == i)[0]
        group_active = not (
            torch.tensor(img_rescaled.reshape(-1, 3))[group_idx.cpu()] == 0
        ).all()
        if group_active:
            idx.append(i)
    idx = torch.tensor(idx)
    if len(idx) < 2:
        return None

    ## Cluster active group embeddings and select n_clusters and visualize as objects
    # Get the group embeddings of the active groups
    groups_emb_img = torch.gather(
        input=groups_emb[log_id].cpu(),
        dim=1,
        index=idx.cpu().unsqueeze(0).repeat((3, 1)),
    ).numpy()

    imgs = []
    colors = torch.tensor(
        [
            (255, 0, 0),  # Red
            (0, 0, 255),  # Blue
            (0, 255, 0),  # Green
            (255, 255, 0),  # Yellow
            (0, 0, 0),  # Black
        ],
        dtype=torch.uint8,
    )
    for i in range(1, 5):
        n_cluster = i
        if len(idx) < 4 and n_cluster > len(idx):
            n_cluster = len(idx)
        ## Color groups by cluster assignment
        # Assign cluster to each group
        k_means = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto")
        cluster_ass = k_means.fit_predict(groups_emb_img.transpose())
        idx_all = torch.ones(groups[log_id].max() + 1, dtype=torch.int64) * (
            len(colors) - 1
        )
        idx_all.scatter_(
            dim=0, index=idx.cpu(), src=torch.tensor(cluster_ass, dtype=torch.int64)
        )

        # Assign color_idx to each group in 224 x 224 grid
        group_colors_img_idx = torch.gather(
            input=idx_all, dim=0, index=groups[log_id].reshape(-1).cpu()
        )
        group_colors_img_idx = group_colors_img_idx.unsqueeze(-1).repeat(1, 3)

        # Insert color for each pixel
        group_colors_img = torch.gather(
            input=colors,
            index=group_colors_img_idx,
            dim=0,
        )
        group_colors_img = group_colors_img.reshape_as(torch.tensor(masked_img))
        imgs.append(group_colors_img.numpy())

    # Create a 2x2 grid from imgs
    combined_img_top = np.concatenate((imgs[0], imgs[1]), axis=1)
    combined_img_bottom = np.concatenate((imgs[2], imgs[3]), axis=1)
    combined_img = np.concatenate((combined_img_top, combined_img_bottom), axis=1)

    # Create a 2x2 grid of the masked image
    tiled_masked_img = np.tile(img_rescaled, (1, 4, 1))
    superimposed_emb_grouped = cv2.addWeighted(
        tiled_masked_img, 0.5, combined_img, 0.5, 0
    )

    # Stack the masked image and the superimposed image
    combined_img = np.concatenate((superimposed_emb_grouped, combined_img), axis=0)

    wandb.log(
        {
            "image_embedding_clustered": [
                wandb.Image(
                    Image.fromarray(combined_img),
                    caption=(
                        f"Clustered embeddings with differening number of clusters, "
                    ),
                    file_type="jpg",
                )
            ],
        }
    )

    return None


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def apply_bounding_box(image, mask):
    """
    Applies the bounding box to the image and masks out all pixels within the bounding box whose value is not True.

    Parameters:
    - image: 2D (grayscale) or 3D (color) numpy array representing the image.
    - mask: 2D boolean numpy array of the same height and width as the image.

    Returns:
    - masked_image: The image with the specified pixels masked out.
    """
    bbox = get_bounding_box(mask)
    if bbox is None:
        # Return an empty image or handle as needed
        return np.zeros_like(image)

    min_row, max_row, min_col, max_col = bbox

    # Extract the ROI from the image and mask
    image_roi = image[min_row : max_row + 1, min_col : max_col + 1]
    mask_roi = mask[min_row : max_row + 1, min_col : max_col + 1]

    # Apply the mask to the ROI
    masked_image = image_roi * np.expand_dims(mask_roi, -1)

    return masked_image


def get_bounding_box(mask):
    """
    Finds the bounding box of all True values in a boolean matrix.

    Parameters:
    - mask: 2D boolean numpy array.

    Returns:
    - (min_row, max_row, min_col, max_col): Coordinates of the bounding box.
    Returns None if there are no True values.
    """
    # Find indices where mask is True
    true_indices = np.argwhere(mask)
    if true_indices.size == 0:
        # No True values found
        return None

    # Get the bounding box coordinates
    min_row, min_col = true_indices.min(axis=0)
    max_row, max_col = true_indices.max(axis=0)

    return min_row, max_row, min_col, max_col


def compute_maskiou(mask1, mask2):
    intersection = (mask1 * mask2).sum((1, 2))
    union = torch.logical_or(mask1, mask2).sum((1, 2)) + 1e-12
    return intersection / union


def compute_pred_correctness(mask_pred, mask_true):
    # Compute the percentage of correctly predicted pixels
    correct = (mask_pred * mask_true).sum((1, 2))
    num_pred = mask_pred.sum((1, 2)) + 1e-12
    return correct / num_pred


class MaskEvaluator:
    """Computes PxAP for a given set of masks."""

    def __init__(self, cam_curve_interval=0.00001):
        # Note: As I understand it, their last two bins are necessary,
        # to go until the end of the curve and compute differences
        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(
            self.cam_threshold_list, [1.0, 2.0, 3.0]
        )
        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=np.float64)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=np.float64)

    def accumulate(self, scoremap, gt_mask):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float64)
            gt_mask: Ground truth.
        """
        scoremap = scoremap.cpu().numpy()
        gt_mask = gt_mask.cpu().numpy()
        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(
            gt_true_scores, bins=self.threshold_list_right_edge
        )
        self.gt_true_score_hist += gt_true_hist.astype(np.float64)

        gt_false_hist, _ = np.histogram(
            gt_false_scores, bins=self.threshold_list_right_edge
        )
        self.gt_false_score_hist += gt_false_hist.astype(np.float64)

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        return auc
