import torch


def compute_contribution_top_feature(features, outputs, weights,  labels):
    with torch.no_grad():
        total_pre_softmax, predicted_classes = torch.max(outputs, dim=1)
        feature_part = features * weights.to(features.device)[predicted_classes]
        class_specific_feature_part = torch.zeros((weights.shape[0], features.shape[1],))
        feature_class_part = torch.zeros((weights.shape[0], features.shape[1],))
        for unique_class in predicted_classes.unique():
            mask = predicted_classes == unique_class
            class_specific_feature_part[unique_class] = feature_part[mask].mean(dim=0)
            gt_mask = labels == unique_class
            feature_class_part[unique_class] = feature_part[gt_mask].mean(dim=0)
        abs_features = feature_part.abs()
        abs_sum = abs_features.sum(dim=1)
        fractions_abs = abs_features / abs_sum[:, None]
        abs_max = fractions_abs.max(dim=1)[0]
        mask = ~torch.isnan(abs_max)
        abs_max = abs_max[mask]
    return abs_max.mean()