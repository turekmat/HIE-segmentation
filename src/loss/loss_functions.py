import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from monai.losses import DiceFocalLoss
from monai.networks.utils import one_hot

def get_loss_function(loss_name, alpha=0, class_weights=None,
                      ft_alpha=0.7, ft_beta=0.3, ft_gamma=4/3,
                      focal_alpha=0.75, focal_gamma=2.0, alpha_mix=0.6,
                      out_channels=2):
    """
    Vrátí ztrátovou funkci podle zadaného jména.
    
    Args:
        loss_name (str): Jméno ztrátové funkce
        alpha (float): Parametr vyvážení pro kombinované ztráty
        class_weights (torch.Tensor): Váhy tříd pro CrossEntropyLoss
        ft_alpha (float): Alpha parametr pro focal tversky loss
        ft_beta (float): Beta parametr pro focal tversky loss
        ft_gamma (float): Gamma parametr pro focal tversky loss
        focal_alpha (float): Alpha parametr pro focal loss
        focal_gamma (float): Gamma parametr pro focal loss
        alpha_mix (float): Parametr míchání pro combined_focal_dice_loss
        out_channels (int): Počet výstupních kanálů
        
    Returns:
        callable: Ztrátová funkce
    """
    if loss_name == "weighted_ce":
        ce_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        def loss_fn(logits, labels):
            return ce_criterion(logits, labels)
        return loss_fn
        
    elif loss_name == "weighted_ce_dice":
        ce_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        def loss_fn(logits, labels):
            return weighted_ce_plus_dice_loss(
                logits=logits,
                labels=labels,
                ce_criterion=ce_criterion,
                alpha=alpha
            )
        return loss_fn

    elif loss_name == "log_cosh_dice":
        def loss_fn(logits, labels):
            return log_cosh_dice_loss(logits, labels)
        return loss_fn

    elif loss_name == "focal_ce_combo":
        ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

        def loss_fn(logits, labels):
            focal_val = focal_loss(logits, labels, alpha=focal_alpha, gamma=focal_gamma)
            ce_val    = ce_criterion(logits, labels)
            return 0.8 * focal_val + 0.25 * ce_val

        return loss_fn

    elif loss_name == "focal_dice_combo":
        def loss_fn(logits, labels):
            return combined_focal_dice_loss(
                logits=logits,
                labels=labels,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                alpha_mix=alpha_mix
            )
        return loss_fn

    elif loss_name == "focal_tversky":
        def loss_fn(logits, labels):
            return focal_tversky_loss(
                logits=logits,
                labels=labels,
                alpha=0.3,
                beta=0.7,
                gamma=2
            )
        return loss_fn

    elif loss_name == "dice_focal":
      dice_focal_loss = DiceFocalLoss(
          include_background=True,
          to_onehot_y=False,
          softmax=True,
          squared_pred=True,
          alpha=focal_alpha,
          gamma=focal_gamma
      )

      def loss_fn(logits, labels):
          labels = one_hot(labels.unsqueeze(1), num_classes=out_channels)
          return dice_focal_loss(logits, labels)

      return loss_fn

    elif loss_name == "focal":
        def loss_fn(logits, labels):
            return focal_loss(
                logits,
                labels,
                alpha=0.8,
                gamma=2.0
            )
        return loss_fn

    elif loss_name == "log_hausdorff":
        def loss_fn(logits, labels):
            return log_hausdorff_loss(logits, labels, alpha=2.0)
        return loss_fn

    else:
        raise ValueError(f"Unsupported loss_name: {loss_name}")


def combined_focal_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    focal_alpha: float = 0.8,
    focal_gamma: float = 2.0,
    alpha_mix:  float = 0.8,
    eps:        float = 1e-6
) -> torch.Tensor:
    """
    Kombinace Focal Loss a Dice Loss.
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        focal_alpha: Alpha parametr pro focal loss
        focal_gamma: Gamma parametr pro focal loss
        alpha_mix: Váha mezi focal a dice loss
        eps: Epsilon pro numerickou stabilitu
        
    Returns:
        torch.Tensor: Výsledná ztráta
    """
    focal_val = focal_loss(
        logits=logits,
        labels=labels,
        alpha=focal_alpha,
        gamma=focal_gamma,
        eps=eps
    )

    dice_val = soft_dice_loss(logits, labels, smooth=1.0)

    combined = alpha_mix * focal_val + (1.0 - alpha_mix) * dice_val
    return combined


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.8,
    gamma: float = 2.0,
    eps:   float = 1e-6
) -> torch.Tensor:
    """
    Implementace Focal Loss pro binární segmentaci.
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        alpha: Váhový faktor pro pozitivní třídu
        gamma: Focusing parametr (redukce well-classified examples)
        eps: Epsilon pro numerickou stabilitu
        
    Returns:
        torch.Tensor: Focal loss
    """
    probs = F.softmax(logits, dim=1)[:, 1, ...]  # shape: [B, D, H, W]

    fg_mask = (labels == 1).float()

    probs = probs.clamp(min=eps, max=1.0 - eps)

    focal_fg = - alpha * fg_mask * ((1.0 - probs) ** gamma) * torch.log(probs)
    focal_bg = - (1.0 - alpha) * (1.0 - fg_mask) * (probs ** gamma) * torch.log(1.0 - probs)

    focal_loss_val = (focal_fg + focal_bg).mean()
    return focal_loss_val


def log_hausdorff_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 2.0,
    eps:   float = 1e-5
) -> torch.Tensor:
    """
    Logarithmická Hausdorff loss funkce.
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        alpha: Parametr pro penalizaci
        eps: Epsilon pro numerickou stabilitu
        
    Returns:
        torch.Tensor: Log-Hausdorff loss
    """
    prob = F.softmax(logits, dim=1)[:, 1, ...]   # tvar [B, D, H, W]
    B = prob.shape[0]

    loss_accum = 0.0
    for i in range(B):
        p_mask_np = (prob[i] >= 0.5).detach().cpu().numpy().astype(np.uint8)
        q_mask_np = labels[i].detach().cpu().numpy().astype(np.uint8)

        dp = distance_transform_edt(1 - p_mask_np)
        dq = distance_transform_edt(1 - q_mask_np)

        diff = (p_mask_np - q_mask_np)**2

        term_map = diff * ((dp**alpha) + (dq**alpha))
        mean_term = term_map.mean()

        loss_i = np.log(1.0 + mean_term + eps)
        loss_accum += loss_i

    loss_value = loss_accum / B
    return torch.tensor(loss_value, dtype=torch.float, requires_grad=True).to(logits.device)


def focal_tversky_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.7,
    beta: float  = 0.3,
    gamma: float = 4/3,
    smooth: float= 1.0
) -> torch.Tensor:
    """
    Focal Tversky Loss - varianta dice loss s penalizací FP a FN a focal weighting.
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        alpha: Váha pro false negatives
        beta: Váha pro false positives
        gamma: Focal exponent
        smooth: Smooth faktor pro výpočet indexu
        
    Returns:
        torch.Tensor: Focal Tversky Loss
    """
    probs = F.softmax(logits, dim=1)
    fg_prob = probs[:, 1, ...]

    fg_label = (labels == 1).float()

    dims = (1,2,3)
    tp = (fg_prob * fg_label).sum(dim=dims)
    fp = ((1.0 - fg_label) * fg_prob).sum(dim=dims)
    fn = (fg_label * (1.0 - fg_prob)).sum(dim=dims)

    tversky_index = (tp + smooth) / (tp + alpha*fn + beta*fp + smooth)

    ftl = (1.0 - tversky_index) ** gamma

    return ftl.mean()


def soft_dice_loss(logits, labels, smooth=1.0):
    """
    Soft Dice Loss pro binární segmentaci.
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        smooth: Smooth faktor pro výpočet Dice koeficientu
        
    Returns:
        torch.Tensor: 1 - Dice koeficient
    """
    prob = F.softmax(logits, dim=1)
    fg   = prob[:,1]
    target_fg = (labels == 1).float()
    inter = (fg * target_fg).sum()
    union= fg.sum() + target_fg.sum()
    dice = (2.*inter + smooth)/(union + smooth)
    return 1.0 - dice


def dice_coefficient(pred, labels, smooth=1.0):
    """
    Vypočítá Dice koeficient mezi predikcí a referencí.
    
    Args:
        pred: Predikce (numpy array)
        labels: Reference (numpy array)
        smooth: Smooth faktor
        
    Returns:
        float: Dice koeficient (0-1)
    """
    pred_fg = (pred == 1).astype(np.float32)
    target_fg = (labels == 1).astype(np.float32)

    inter = (pred_fg * target_fg).sum()
    union = pred_fg.sum() + target_fg.sum()
    dice = (2. * inter + smooth) / (union + smooth)

    return dice


def weighted_ce_plus_dice_loss(logits, labels, ce_criterion, alpha=0.5):
    """
    Weighted Cross Entropy + alpha * Soft Dice Loss
    
    Args:
        logits: Model output logits
        labels: Ground truth labels
        ce_criterion: Cross-entropy loss functor
        alpha: Váha pro Dice loss
        
    Returns:
        torch.Tensor: Kombinovaná ztráta
    """
    ce_loss = ce_criterion(logits, labels)     # Weighted CE
    dice_l  = soft_dice_loss(logits, labels)   # 1 - Dice coefficient
    return ce_loss + alpha * dice_l


def log_cosh_dice_loss(logits, labels, smooth=1.0):
    """
    Compute the Log-Cosh Dice Loss.

    Args:
        logits: Model output logits (B, 2, D, H, W)
        labels: Ground truth labels (B, D, H, W)
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        torch.Tensor: Log-Cosh Dice Loss
    """
    # Probabilities via softmax, pick foreground channel
    prob = F.softmax(logits, dim=1)  # [B, 2, D, H, W]
    fg   = prob[:, 1, ...]          # foreground probability map
    target_fg = (labels == 1).float()

    intersection = (fg * target_fg).sum(dim=[1,2,3])   # sum per batch item
    union        = fg.sum(dim=[1,2,3]) + target_fg.sum(dim=[1,2,3])

    dice = (2.0 * intersection + smooth) / (union + smooth)  # per-batch dice
    dice_loss = 1.0 - dice                                   # [B]-shaped

    # Apply log-cosh
    log_cosh = torch.log(torch.cosh(dice_loss))
    # Average across the batch dimension
    return log_cosh.mean()


def compute_loss_per_sample(loss_fn, logits, labels):
    """
    Vypočítá ztrátu pro každý vzorek zvlášť.
    
    Args:
        loss_fn: Ztrátová funkce
        logits: Model output logits 
        labels: Ground truth labels
        
    Returns:
        torch.Tensor: Tensor ztrát pro každý vzorek
    """
    batch_losses = []
    for i in range(logits.shape[0]):
        # Vybereme i-tý vzorek a přidáme dimenzi batch
        sample_loss = loss_fn(logits[i].unsqueeze(0), labels[i].unsqueeze(0))
        batch_losses.append(sample_loss)
    return torch.stack(batch_losses)  # vrátí tensor tvaru [B] 