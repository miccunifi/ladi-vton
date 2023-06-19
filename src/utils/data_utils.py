import torch


def mask_features(features: list, mask: torch.Tensor):
    """
    Mask features with the given mask.
    """

    for i, feature in enumerate(features):
        # Resize the mask to the feature size.
        mask = torch.nn.functional.interpolate(mask, size=feature.shape[-2:])

        # Mask the feature.
        features[i] = feature * (1 - mask)

    return features
