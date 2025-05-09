import os
import torch
from torch import nn
from torchvision import models
from transformers import ViTForImageClassification
import torch.nn.functional as F


def create_model(config):
    """
    Parse the configuration file and return a relevant model
    """
    from models.models import P2P

    if config.model.model == "P2P":
        return P2P(config)
    else:
        print("Could not create model with name ", config.model, "!")
        quit()


def create_classifier(config):
    """Initialize the classifier architecture"""
    classifier_arch = config.model.classifier_arch
    if classifier_arch == "resnet18":
        # Search for weights
        classifier = load_model_torch(models.resnet18, config.model.model_path)

        # Model was pretrained on imagenet, so we need to change the last layer if we use a different dataset
        if config.data.dataset != "imagenet":
            n_features = classifier.fc.in_features
            classifier.fc = nn.Linear(n_features, config.data.num_classes)

    elif classifier_arch == "resnet34":
        # Search for weights
        classifier = load_model_torch(models.resnet34, config.model.model_path)

        # Model was pretrained on imagenet, so we need to change the last layer if we use a different dataset
        if config.data.dataset != "imagenet":
            n_features = classifier.fc.in_features
            classifier.fc = nn.Linear(n_features, config.data.num_classes)

    elif classifier_arch == "vit_base":
        classifier = load_model_hf(
            ViTForImageClassification,
            "google/vit-base-patch16-224",
            config.model.model_path,
            config.data.num_classes,
        )

    elif classifier_arch == "vit_tiny":
        classifier = load_model_hf(
            ViTForImageClassification,
            "WinKawaks/vit-tiny-patch16-224",
            config.model.model_path,
            config.data.num_classes,
        )

    else:
        raise NotImplementedError("ERROR: architecture not supported!")

    return Classifier_Wrapper(classifier)


def create_importance_predictor(config):
    """Initialize the importance predictor architecture"""
    importance_predictor_arch = config.model.importance_predictor_arch
    num_dims = config.model.get("num_dims", 1)
    if importance_predictor_arch == "mobilenet":
        model = load_model_torch(
            models.segmentation.lraspp_mobilenet_v3_large,
            config.model.model_path,
        )
        model.classifier.low_classifier = nn.Conv2d(
            40, num_dims, kernel_size=(1, 1), stride=(1, 1)
        )
        model.classifier.high_classifier = nn.Conv2d(
            128, num_dims, kernel_size=(1, 1), stride=(1, 1)
        )
        if config.model.use_dynamic_threshold:
            model.conv_input = {}

            def get_conv_input(name):
                def hook(module, input):
                    model.conv_input[name] = input[0]

                return hook

            model.classifier.low_classifier.register_forward_pre_hook(
                get_conv_input("low")
            )
            model.classifier.high_classifier.register_forward_pre_hook(
                get_conv_input("high")
            )
        importance_predictor = Wrapper(model)
    else:
        raise NotImplementedError("ERROR: architecture not supported!")

    return importance_predictor


def load_model_torch(arch, model_directory):
    """load model from torch hub"""
    os.environ["TORCH_HOME"] = model_directory
    for root, dirs, files in os.walk(model_directory):
        for file in files:
            if arch.__name__ in file:
                # Get full path of file
                model_path = os.path.join(root, file)

                # Load the model
                model = arch(weights=None)
                model.load_state_dict(torch.load(model_path))
                return model

    else:
        print(f"Weights not found at {model_directory}. Downloading...")
        model = arch(weights="DEFAULT")
        return model


def load_model_hf(arch, version, model_directory, num_classes=None):
    """load model from huggingface"""

    for root, dirs, files in os.walk(model_directory):
        for file in files:
            if version in file:
                # Get full path of file
                model_path = os.path.join(root, file)

                # Load the model
                model = arch.from_pretrained(
                    model_path, num_labels=num_classes, ignore_mismatched_sizes=True
                )

                return model

            elif version in root:
                if num_classes is None:
                    # Used for importance predictor if vit base, where we only need the hidden states
                    model = arch.from_pretrained(root, add_pooling_layer=False)
                else:
                    # # Importance Predictor & Classifier without pretraining. Note that num_classes is fixed to what was used prior, and likely NO ERROR is thrown if there's a mismatch
                    # from transformers import AutoConfig

                    # config = AutoConfig.from_pretrained(root, local_files_only=True)
                    # model = arch(config)
                    model = arch.from_pretrained(
                        root, num_labels=num_classes, ignore_mismatched_sizes=True
                    )
                return model
    else:
        print(f"Weights not found at {model_directory}. Downloading...")
        # Storing the model with default number of classes
        model = arch.from_pretrained(version)
        # Create a directory with the name "version" in model_directory
        os.makedirs(model_directory + version, exist_ok=True)
        model.save_pretrained(model_directory + version)

        if num_classes is None:
            # Used for importance predictor if vit base, where we only need the hidden states
            model = arch.from_pretrained(root, add_pooling_layer=False)
        else:
            # Loading model with correct number of classes
            model = arch.from_pretrained(
                model_directory + version,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

        return model


class Wrapper(nn.Module):
    """This module takes the model and in the forward pass directly extracts "out" in the output dict"""

    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]

    def threshold_as_input(self, low_input, high_input, out_shape):
        """hook to provide the sampled threshold τ as input to the model,
        such that it can adapt its masking to the desired level of sparsity.
        This is specific to MobileNet, for other models it should be sufficient to
        just add it to the penultimate representation layer and continue with the forward pass.
        """
        # This is the full model forward pass after the hook
        low_output = self.model.classifier.low_classifier(low_input)
        high_output = self.model.classifier.high_classifier(high_input)
        out = low_output + high_output
        out = F.interpolate(out, size=out_shape, mode="bilinear", align_corners=False)

        return out


class Classifier_Wrapper(nn.Module):
    """Wrapper for the classifier model to get logits directly from the model output
    This is needed for the ViT model, as it has a different output format"""

    def __init__(self, model):
        super(Classifier_Wrapper, self).__init__()
        self.model = model
        self.get_logits = isinstance(model, ViTForImageClassification)

    def forward(self, x):
        if self.get_logits:
            return self.model(x).logits
        else:
            return self.model(x)
