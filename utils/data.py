import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from fast_slic import Slic
import numpy as np
import pickle


def get_data(config_data, gen):
    """
    Parse the config_data file and return a relevant dataset
    """
    assert (
        config_data.resolution == 224
    ), "Only 224x224 resolution is supported for now, to work with ViTs"
    if config_data.dataset == "imagenet":
        trainset, validset, testset = get_ImageNet_dataloader(
            config_data.data_path,
            use_superpixels=config_data.precompute_superpixels,
            color_aug_intensity=config_data.color_aug_intensity,
            n_segments=config_data.n_segments,
            compactness=config_data.compactness,
        )

    elif config_data.dataset == "imagenet-9":
        trainset, validset, testset = get_ImageNet9_dataloader(
            config_data.data_path,
            use_superpixels=config_data.precompute_superpixels,
            color_aug_intensity=config_data.color_aug_intensity,
            n_segments=config_data.n_segments,
            compactness=config_data.compactness,
        )

    elif config_data.dataset == "cifar10":
        print("CIFAR-10 DATASET")
        trainset, validset, testset = get_CIFAR10_dataset(
            config_data.data_path,
            config_data.resolution,
            use_superpixels=config_data.precompute_superpixels,
            color_aug_intensity=config_data.color_aug_intensity,
            n_segments=config_data.n_segments,
            compactness=config_data.compactness,
        )
    elif config_data.dataset == "bam":
        print("BAM DATASET")
        trainset, validset, testset = get_BAM_dataset(
            config_data.data_path,
            target=config_data.target,
            use_superpixels=config_data.precompute_superpixels,
            color_aug_intensity=config_data.color_aug_intensity,
            n_segments=config_data.n_segments,
            compactness=config_data.compactness,
        )
    elif config_data.dataset == "coco":
        print("COCO DATASET")
        trainset, validset, testset = get_COCO_dataset(
            config_data.data_path,
            use_superpixels=config_data.precompute_superpixels,
            color_aug_intensity=config_data.color_aug_intensity,
            n_segments=config_data.n_segments,
            compactness=config_data.compactness,
        )
    else:
        raise NotImplementedError("ERROR: Dataset not supported!")

    train_loader = DataLoader(
        trainset,
        batch_size=config_data.batch_size,
        shuffle=True,
        num_workers=config_data.workers,
        pin_memory=True,
        generator=gen,
        drop_last=True,
    )
    val_loader = DataLoader(
        validset,
        batch_size=config_data.batch_size,
        shuffle=False,
        num_workers=config_data.workers,
        pin_memory=True,
        generator=gen,
    )
    test_loader = DataLoader(
        testset,
        batch_size=config_data.batch_size,
        shuffle=False,
        num_workers=config_data.workers,
        generator=gen,
    )

    return train_loader, val_loader, test_loader


def get_normalization_transform(dataset):
    if dataset in ["imagenet", "cifar10", "imagenet-9", "bam", "coco"]:
        return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif dataset is None:
        print("No dataset specified. Using ImageNet normalization.")
        return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        raise NotImplementedError("ERROR: Dataset not supported!")


def get_ImageNet_dataloader(
    datapath,
    use_superpixels=False,
    color_aug_intensity=0,
    n_segments=100,
    compactness=10,
):
    datapath = datapath + "imagenet"

    traindir = os.path.join(datapath, "train")
    valdir = os.path.join(datapath, "val")

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    test_transform = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]

    train_transform, test_transform = get_color_superpixel_norm_transforms(
        train_transform,
        test_transform=test_transform,
        dataset="imagenet",
        use_superpixels=use_superpixels,
        color_aug_intensity=color_aug_intensity,
        n_segments=n_segments,
        compactness=compactness,
    )

    data_transforms = {
        "train": transforms.Compose(train_transform),
        "test": transforms.Compose(test_transform),
    }

    trainset = datasets.ImageFolder(traindir, data_transforms["train"])
    validset = datasets.ImageFolder(valdir, data_transforms["test"])
    testset = datasets.ImageFolder(valdir, data_transforms["test"])

    return trainset, validset, testset


def get_ImageNet9_dataloader(
    datapath,
    use_superpixels=False,
    color_aug_intensity=0,
    n_segments=100,
    compactness=10,
):
    # Requires imagenet dataset to be downloaded and stored in the datapath,
    # as well as imagenet9_train_orig_path.pkl & imagenet9_val_orig_path.pkl for train&val creation of imagenet9,
    # as well as backgrounds_challenge_data from https://github.com/MadryLab/backgrounds_challenge/releases stored in the datapath for test set

    # Create Imagenet9 dataset from Imagenet if doesn't already exist
    id_to_label = {
        0: "dog",
        1: "bird",
        2: "wheeled_vehicle",
        3: "reptile",
        4: "carnivore",
        5: "insect",
        6: "musical_instrument",
        7: "primate",
        8: "fish",
    }
    if not os.path.isdir(os.path.join(datapath, "imagenet9", "train")):
        subset_train = open(
            os.path.join(datapath, "imagenet9", "imagenet9_train_orig_path.pkl"),
            "rb",
        )
        data_addresses_train = pickle.load(subset_train)
        subset_train.close()
        os.makedirs(os.path.join(datapath, "imagenet9", "train"))

        for i, category in enumerate(data_addresses_train):
            if not os.path.isdir(
                os.path.join(datapath, "imagenet9", "train", f"0{i}_{id_to_label[i]}")
            ):
                os.makedirs(
                    os.path.join(
                        datapath, "imagenet9", "train", f"0{i}_{id_to_label[i]}"
                    )
                )
            for j, file_path in enumerate(category):
                file_new_path = os.path.join(
                    "imagenet9",
                    "train",
                    f"0{i}_{id_to_label[i]}",
                    file_path.split("/")[-1],
                )
                os.system(
                    f"cp {os.path.join(datapath + file_path)} {os.path.join(datapath + file_new_path)}"
                )

    if not os.path.isdir(os.path.join(datapath, "imagenet9", "val")):
        subset_val = open(
            os.path.join(datapath, "imagenet9", "imagenet9_val_orig_path.pkl"),
            "rb",
        )
        data_addresses_val = pickle.load(subset_val)
        subset_val.close()
        os.makedirs(os.path.join(datapath, "imagenet9", "val"))

        for i, category in enumerate(data_addresses_val):
            if not os.path.isdir(
                os.path.join(datapath, "imagenet9", "val", f"0{i}_{id_to_label[i]}")
            ):
                os.makedirs(
                    os.path.join(datapath, "imagenet9", "val", f"0{i}_{id_to_label[i]}")
                )
            for j, file_path in enumerate(category):
                file_new_path = os.path.join(
                    "imagenet9",
                    "val",
                    f"0{i}_{id_to_label[i]}",
                    file_path.split("/")[-1],
                )
                os.system(
                    f"cp {os.path.join(datapath + file_path)} {os.path.join(datapath + file_new_path)}"
                )

    datapath = datapath + "imagenet9"

    traindir = os.path.join(datapath, "train")
    valdir = os.path.join(datapath, "val")
    testdir = os.path.join(
        datapath,
        "backgrounds_challenge_data",
        "bg_challenge",
        "original",
        "val",
    )
    testdir_segmentation = os.path.join(
        datapath,
        "backgrounds_challenge_data",
        "bg_challenge",
        "fg_mask",
        "val",
    )

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    val_transform = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    # IMPORTANT: Test set has already been resized to 224x224 by the Imagenet9 authors.
    # It is important to retain this, such that the segmentation map corresponds to the correct image.
    test_transform = [
        transforms.ToTensor(),
    ]

    train_transform, val_transform, test_transform = (
        get_color_superpixel_norm_transforms(
            train_transform,
            val_transform,
            test_transform,
            dataset="imagenet-9",
            use_superpixels=use_superpixels,
            color_aug_intensity=color_aug_intensity,
            n_segments=n_segments,
            compactness=compactness,
        )
    )

    data_transforms = {
        "train": transforms.Compose(train_transform),
        "val": transforms.Compose(val_transform),
        "test": transforms.Compose(test_transform),
    }

    trainset = datasets.ImageFolder(traindir, data_transforms["train"])
    validset = datasets.ImageFolder(valdir, data_transforms["val"])
    testset_base = datasets.ImageFolder(testdir, data_transforms["test"])

    # Including ground-truth segmentation mask for the test set
    testset = SegmentationDataset(testset_base, testdir_segmentation)

    return trainset, validset, testset


def get_CIFAR10_dataset(
    datapath,
    resolution,
    use_superpixels=False,
    color_aug_intensity=0,
    n_segments=100,
    compactness=10,
):
    datapath = datapath + "cifar10/"

    train_transform = [
        transforms.RandomHorizontalFlip(),  # Random horizontal flip with a probability of 0.5
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    ]

    test_transform = [
        transforms.ToTensor(),
    ]

    if resolution == 224:  # For working with ViTs
        print("Using 224x224 resolution to work with ViTs")
        train_transform.insert(0, transforms.Resize(256))
        test_transform.insert(0, transforms.Resize(256))
        train_transform.insert(1, transforms.RandomCrop(224))
        test_transform.insert(1, transforms.CenterCrop(224))

    train_transform, test_transform = get_color_superpixel_norm_transforms(
        train_transform,
        test_transform=test_transform,
        dataset="cifar10",
        use_superpixels=use_superpixels,
        color_aug_intensity=color_aug_intensity,
        n_segments=n_segments,
        compactness=compactness,
    )

    data_transforms = {
        "train": transforms.Compose(train_transform),
        "test": transforms.Compose(test_transform),
    }

    trainset = datasets.CIFAR10(
        root=datapath, train=True, download=True, transform=data_transforms["train"]
    )

    validset = datasets.CIFAR10(
        root=datapath, train=False, download=True, transform=data_transforms["test"]
    )

    testset = datasets.CIFAR10(
        root=datapath, train=False, download=True, transform=data_transforms["test"]
    )

    return trainset, validset, testset


def get_BAM_dataset(
    datapath,
    target,
    use_superpixels=False,
    color_aug_intensity=0,
    n_segments=100,
    compactness=10,
):
    datapath = os.path.join(datapath, "bam", "data")
    if target == "scene":
        datapath = os.path.join(datapath, "scene")
    elif target == "object":
        datapath = os.path.join(datapath, "obj")
    else:
        raise NotImplementedError("ERROR: Target not supported!")

    traindir = os.path.join(datapath, "train")
    testdir = os.path.join(datapath, "val")
    testdir_segmentation = os.path.join(datapath, "val_mask")

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    # Cropping test images to 224x224. Note that segmentation mask has already been center cropped to 224x224 BAM authors.
    test_transform = [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]

    train_transform, test_transform = get_color_superpixel_norm_transforms(
        train_transform,
        test_transform=test_transform,
        dataset="bam",
        use_superpixels=use_superpixels,
        color_aug_intensity=color_aug_intensity,
        n_segments=n_segments,
        compactness=compactness,
    )

    data_transforms = {
        "train": transforms.Compose(train_transform),
        "test": transforms.Compose(test_transform),
    }

    trainset = datasets.ImageFolder(traindir, data_transforms["train"])
    validset = datasets.ImageFolder(testdir, data_transforms["test"])
    testset_base = datasets.ImageFolder(testdir, data_transforms["test"])

    # Including ground-truth segmentation mask for the test set
    testset = SegmentationDataset(testset_base, testdir_segmentation)

    return trainset, validset, testset


def get_COCO_dataset(
    datapath,
    use_superpixels=False,
    color_aug_intensity=0,
    n_segments=100,
    compactness=10,
):
    datapath = os.path.join(datapath, "coco")
    traindir = os.path.join(datapath, "train")
    testdir = os.path.join(datapath, "val")
    testdir_segmentation = os.path.join(datapath, "mask", "val")

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    # Validation image and segmentation mask have already been resized to 224x224 by us.
    # We have done this preprocessing s.t. no errors can be introduced by the user to the data.
    test_transform = [
        transforms.ToTensor(),
    ]

    train_transform, test_transform = get_color_superpixel_norm_transforms(
        train_transform,
        test_transform=test_transform,
        dataset="coco",
        use_superpixels=use_superpixels,
        color_aug_intensity=color_aug_intensity,
        n_segments=n_segments,
        compactness=compactness,
    )

    data_transforms = {
        "train": transforms.Compose(train_transform),
        "test": transforms.Compose(test_transform),
    }

    trainset = datasets.ImageFolder(traindir, data_transforms["train"])
    validset = datasets.ImageFolder(testdir, data_transforms["test"])
    testset_base = datasets.ImageFolder(testdir, data_transforms["test"])

    # Including ground-truth segmentation mask for the test set
    testset = SegmentationDataset(testset_base, testdir_segmentation)

    return trainset, validset, testset


class SLICSuperpixelsTransform:
    """Custom transformation that returns the image and its SLIC segmentation mask"""

    def __init__(self, n_segments=100, compactness=10):
        self.n_segments = n_segments
        self.compactness = compactness
        self.slic = Slic(
            num_components=self.n_segments,
            compactness=self.compactness,
        )

    def __call__(self, image):
        # Apply SLIC superpixel segmentation using FastSLIC(https://github.com/Algy/fast-slic)
        segments = self.slic.iterate(
            np.ascontiguousarray(
                np.moveaxis(np.array(image * 255, dtype=np.uint8), 0, -1)
            )
        )

        # Convert segmentation to a tensor (to match PyTorch workflow)
        segmentation_tensor = torch.tensor(segments, dtype=torch.int64)

        # Return the original image and segmentation mask
        return image, segmentation_tensor


def get_color_superpixel_norm_transforms(
    train_transform,
    val_transform=None,
    test_transform=None,
    dataset=None,
    use_superpixels=False,
    color_aug_intensity=0,
    n_segments=100,
    compactness=10,
):
    """Data preprocessing pipeline for training and testing."""
    # Add color augmentation to the training set
    if color_aug_intensity > 0.0:
        assert color_aug_intensity <= 1.0
        train_transform.append(
            transforms.ColorJitter(
                brightness=0.8 * color_aug_intensity,
                contrast=0.8 * color_aug_intensity,
                saturation=0.8 * color_aug_intensity,
                hue=0.4 * color_aug_intensity,
            )
        )

    normalize_transform = get_normalization_transform(dataset)

    if use_superpixels:
        # Superpixel segmentation
        slic_transform = SLICSuperpixelsTransform(
            n_segments=n_segments, compactness=compactness
        )
        # Normalization
        slic_normalize_transform = transforms.Lambda(
            lambda img_seg: (
                normalize_transform(img_seg[0]),
                img_seg[1],
            )
        )
        if val_transform is not None:
            val_transform.extend([slic_transform, slic_normalize_transform])
        train_transform.extend([slic_transform, slic_normalize_transform])
        test_transform.extend([slic_transform, slic_normalize_transform])
    else:
        # Wrap normalize_transform in a lambda to always return a tuple with a None placeholder.
        normalization_transform = transforms.Lambda(
            lambda img: (normalize_transform(img), torch.tensor([]))
        )

        train_transform.append(normalization_transform)
        test_transform.append(normalization_transform)
        if val_transform is not None:
            val_transform.append(normalization_transform)

    if val_transform is not None:
        return train_transform, val_transform, test_transform
    else:
        return train_transform, test_transform


class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset that loads and processes images, as well as
    their corresponding ground-truth segmentation masks."""

    def __init__(self, base_dataset, segmentation_dir):
        self.base_dataset = base_dataset

        self.segmentation_dataset = datasets.DatasetFolder(
            segmentation_dir, loader=np.load, extensions=("npy")
        )
        self.has_segmentation = True

    def __getitem__(self, idx):
        image = self.base_dataset[idx]
        segmentation = self.segmentation_dataset[idx]
        assert image[1] == segmentation[1]  # Assert same label
        segmentation = torch.tensor(segmentation[0])
        return image + (segmentation,)

    def __len__(self):
        return len(self.base_dataset)


def get_id_to_label(data_config):
    """
    Dictionary that maps class IDs to labels based on the dataset; used for logging.
    """
    if data_config.dataset == "imagenet-9":
        id_to_label = {
            0: "dog",
            1: "bird",
            2: "wheeled_vehicle",
            3: "reptile",
            4: "carnivore",
            5: "insect",
            6: "musical_instrument",
            7: "primate",
            8: "fish",
        }
    elif data_config.dataset == "cifar10":
        id_to_label = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }
    elif data_config.dataset == "bam":
        if data_config.target == "scene":
            id_to_label = {
                0: "bamboo_forest",
                1: "bedroom",
                2: "bowling_alley",
                3: "bus_interior",
                4: "cockpit",
                5: "corn_field",
                6: "laundromat",
                7: "runway",
                8: "ski_slope",
                9: "running_track",
            }
        elif data_config.target == "object":
            id_to_label = {
                0: "backpack",
                1: "bird",
                2: "dog",
                3: "elephant",
                4: "kite",
                5: "pizza",
                6: "stop sign",
                7: "toilet",
                8: "truck",
                9: "zebra",
            }
    elif data_config.dataset == "coco":
        id_to_label = {
            0: "bed",
            1: "car",
            2: "cat",
            3: "clock",
            4: "dog",
            5: "person",
            6: "sink",
            7: "train",
            8: "tv",
            9: "vase",
        }
    elif data_config.dataset == "imagenet":
        # Use ImageNet class labels
        with open("utils/imagenet_classes.txt") as f:
            id_to_label = {
                int(line.split(",")[0]): line.split(",")[1].strip() for line in f
            }
    else:
        # Use config.data.num_classes to create id_to_label
        id_to_label = {i: str(i) for i in range(data_config.num_classes)}
    return id_to_label
