import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss implementation.
        Args:
            alpha (float or list): Class weight. If list, it should have same length as the number of classes.
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities for the correct class
        pt = torch.exp(-ce_loss)

        # Focal loss calculation
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class Similarity_Distribution_Matching_Loss(nn.Module):
    """
    Similarity Distribution Matching (SDM) Loss,
    Adapted from: https://github.com/anosorae/IRRA
    """

    def __init__(self, length):
        super(Similarity_Distribution_Matching_Loss, self).__init__()
        self.length = length

    def forward(self, vision_fetures, text_fetures, labels, epsilon=1e-8):
        logit_scale = self.length
        labels = labels - labels.t()
        labels = (labels == 0).float()

        image_norm = vision_fetures / vision_fetures.norm(dim=1, keepdim=True)
        text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        vision_proj_text = logit_scale * i2t_cosine_theta

        # normalize the true matching distribution
        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(vision_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(vision_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return loss

class joint_loss(nn.Module):
    def __init__(self, w1=0.2, w2=0.01):
        super(joint_loss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.Cross_Entropy_Loss = nn.CrossEntropyLoss()
        self.Similarity_Distribution_Matching_Loss_1 = Similarity_Distribution_Matching_Loss(2)
        self.Similarity_Distribution_Matching_Loss_2 = Similarity_Distribution_Matching_Loss(2)
        self.Similarity_Distribution_Matching_Loss_3 = Similarity_Distribution_Matching_Loss(2)
        self.Focal_loss = FocalLoss()
    def forward(self, modality1_features, modality2_features, modality3_features, labels, scores):
        w1 = self.w1
        w2 = self.w2
        modality1_features = modality1_features.squeeze()
        modality2_features = modality2_features.squeeze()
        modality3_features = modality3_features.squeeze()
        cross_entropy_loss = self.Focal_loss(scores, labels)
        # cross_entropy_loss = self.Cross_Entropy_Loss(scores, labels)

        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        rv_sdm = self.Similarity_Distribution_Matching_Loss_1(modality1_features, modality2_features, labels)
        rc_sdm = self.Similarity_Distribution_Matching_Loss_2(modality2_features, modality3_features, labels)
        vc_sdm = self.Similarity_Distribution_Matching_Loss_3(modality1_features, modality3_features, labels)

        multi_loss = (1 - w1) * rv_sdm + w1 * (rc_sdm + vc_sdm) / 2
        task_loss = cross_entropy_loss
        return task_loss + w2 * multi_loss

        # SDM_loss = self.Similarity_Distribution_Matching_Loss_1(modality1_features, modality2_features, labels)+\
        #             self.Similarity_Distribution_Matching_Loss_2(modality2_features, modality3_features, labels)+\
        #              self.Similarity_Distribution_Matching_Loss_3(modality1_features, modality3_features, labels)

        # return cross_entropy_loss + 0.01 * SDM_loss


