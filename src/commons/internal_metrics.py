"""Threshold dependent metrics"""
import torch
import numpy as np
import skimage
import logging



def get_pred_mask(preds: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    preds: torch.Size([1000, 1000]), torch.float64, [0, 1]
    """
    return preds > threshold


def get_real_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: torch.Size([1000, 1000]), torch.uint8, [0, 255]
    """
    return mask > int(255/2)


def get_real_mask_bin(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: torch.Size([1000, 1000]), torch.uint8, [0, 1]
    """
    return mask > 0.5


def get_tp(pred_mask: torch.Tensor, real_mask:torch.Tensor) -> torch.Tensor:
    """
    pred_mask: torch.Size([1000, 1000]), torch.bool, [True, False]
    real_mask: torch.Size([1000, 1000]), torch.bool, [True, False]
    """
    return np.logical_and(pred_mask, real_mask)


def get_tn(pred_mask: torch.Tensor, real_mask:torch.Tensor) -> torch.Tensor:
    """
    pred_mask: torch.Size([1000, 1000]), torch.bool, [True, False]
    real_mask: torch.Size([1000, 1000]), torch.bool, [True, False]
    """
    return np.logical_and(~pred_mask, ~real_mask)

def get_xor_mask(pred_mask: torch.Tensor, real_mask:torch.Tensor) -> torch.Tensor:
    """
    pred_mask: torch.Size([1000, 1000]), torch.bool, [True, False]
    real_mask: torch.Size([1000, 1000]), torch.bool, [True, False]
    """
    return np.logical_xor(pred_mask, real_mask)

def get_fp(pred_mask: torch.Tensor, xor_mask:torch.Tensor):
    """
    pred_mask: torch.Size([1000, 1000]), torch.bool, [True, False]
    xor_mask: torch.Size([1000, 1000]), torch.uint8, [0, 1]
    """
    return np.logical_and(pred_mask, xor_mask)

def get_fn(real_mask: torch.Tensor, xor_mask:torch.Tensor):
    """
    real_mask: torch.Size([1000, 1000]), torch.bool, [True, False]
    xor_mask: torch.Size([1000, 1000]), torch.uint8, [0, 1]
    """
    return np.logical_and(real_mask, xor_mask)


def fpr_calculation(fp, tn):
    if fp == 0:
        return 0
    elif fp + tn == 0:
        return None
    else: 
        return fp / (fp + tn)


def prc_calculation(tp, fp):
    if tp == 0:
        return 0
    elif fp + tp == 0:
        return None
    else: 
        return tp / (fp + tp)


def intersection_over_union_calculation(tp, fp, fn):
    if tp == 0:
        return 0
    elif fp + tp + fn == 0:
        return None
    else: 
        return tp / (fp + tp + fn)


def get_fpr_of_mask(true_mask: np.ndarray, score_mask: np.ndarray) -> float:
    # False positive rate
    mask_tn = get_tn(score_mask, true_mask)
    mask_xor = get_xor_mask(score_mask, true_mask)
    mask_fp = get_fp(score_mask, mask_xor)
    FP = mask_fp.sum()
    TN = mask_tn.sum()
    return fpr_calculation(fp=FP, tn=TN)
    

def get_fpr(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    # False positive rate
    true_mask = get_real_mask_bin(y_true)
    score_mask = get_pred_mask(y_score, threshold)
    return get_fpr_of_mask(true_mask, score_mask)


def get_prc_of_mask(true_mask: np.ndarray, score_mask: np.ndarray) -> float:
    # Precision
    mask_tp = get_tp(score_mask, true_mask)
    mask_xor = get_xor_mask(score_mask, true_mask)
    mask_fp = get_fp(score_mask, mask_xor)
    FP = mask_fp.sum()
    TP = mask_tp.sum()
    return prc_calculation(tp=TP, fp=FP)


def get_prc(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    # Precision
    true_mask = get_real_mask_bin(y_true)
    score_mask = get_pred_mask(y_score, threshold)
    return get_prc_of_mask(true_mask, score_mask)


def get_iou_of_mask(true_mask: np.ndarray, score_mask: np.ndarray) -> float:
    # Intersection over union
    mask_tp = get_tp(score_mask, true_mask)
    mask_xor = get_xor_mask(score_mask, true_mask)
    mask_fp = get_fp(score_mask, mask_xor)
    mask_fn = get_fn(true_mask, mask_xor)
    FP = mask_fp.sum()
    TP = mask_tp.sum()
    FN = mask_fn.sum()
    return intersection_over_union_calculation(tp=TP, fp=FP, fn=FN)


def get_iou(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    # Intersection over union
    true_mask = get_real_mask_bin(y_true)
    score_mask = get_pred_mask(y_score, threshold)
    return get_iou_of_mask(true_mask, score_mask)


def get_connected_regions(x_mask: torch.Tensor):
    return skimage.measure.label(x_mask, background=False, connectivity=2)


def get_pro_of_mask(true_mask: np.ndarray, score_mask: np.ndarray) -> float:
    # Per region overlap
    pred_conn_components = get_connected_regions(score_mask)
    real_conn_components = get_connected_regions(true_mask)

    max_regions = max(pred_conn_components.max(), real_conn_components.max())
    logging.debug(f"There are a maximum of {max_regions} connected components between predictions and real masks")

    pro = 0
    # for all connected components 
    for conn in range(real_conn_components.max()):
        conn_label = conn + 1 # python starts in index 0
        c_ik_conn_mask = np.where(real_conn_components == conn_label, True, False)
        c_ik_conn_area = len(c_ik_conn_mask[c_ik_conn_mask == True])
        logging.debug(f"C_i,k$: {c_ik_conn_area}")
        
        union_components_conn = np.logical_and(pred_conn_components, c_ik_conn_mask)
        p_i_c_ik_conn_area = len(union_components_conn[union_components_conn == True])
        logging.debug(f"P_i AND C_i,k$: {p_i_c_ik_conn_area}")
        
        pro += p_i_c_ik_conn_area/c_ik_conn_area
    if real_conn_components.max() == 0:
        logging.warning(f"PRO: 0, with real_conn_components.max() = {real_conn_components.max()}")
        return 0
    else:
        pro = pro / (real_conn_components.max())
        logging.debug(f"PRO: {pro:.4f}")
        return pro
    

def get_pro(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    # Per region overlap
    true_mask = get_real_mask_bin(y_true)
    score_mask = get_pred_mask(y_score, threshold)
    return get_pro_of_mask(true_mask, score_mask)
    

def get_precision_overlap_of_mask(true_mask: np.ndarray, score_mask: np.ndarray) -> float:
    # looking at false positives in connected anomalies
    pred_conn_components = get_connected_regions(score_mask)
    real_conn_components = get_connected_regions(true_mask)

    max_regions = max(pred_conn_components.max(), real_conn_components.max())
    logging.debug(f"There are a maximum of {max_regions} connected components between predictions and real masks")

    precision_overlap = 0
    # for all connected components 
    for conn in range(pred_conn_components.max()):
        conn_label = conn + 1 # python starts in index 0
        c_ik_conn_mask = np.where(real_conn_components == conn_label, True, False)
        p_i_conn_mask = np.where(pred_conn_components == conn_label, True, False)
        p_i_conn_area = len(p_i_conn_mask[p_i_conn_mask == True])
        logging.debug(f"P_i: {p_i_conn_area}")
        
        union_components_conn = np.logical_and(pred_conn_components, c_ik_conn_mask)
        p_i_c_ik_conn_area = len(union_components_conn[union_components_conn == True])
        logging.debug(f"P_i AND C_i,k: {p_i_c_ik_conn_area}")
        
        precision_overlap += p_i_c_ik_conn_area/p_i_conn_area
    
    if pred_conn_components.max() == 0:
        logging.warning(f"PRO: 0, with pred_conn_components.max() = {pred_conn_components.max()}")
        return 0
    else:
        fp_overlap = precision_overlap / (pred_conn_components.max())
        logging.debug(f"Precision-overlap: {fp_overlap:.4f}")
        return fp_overlap


def get_precision_overlap(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    # looking at false positives in connected anomalies
    true_mask = get_real_mask_bin(y_true)
    score_mask = get_pred_mask(y_score, threshold)
    return get_precision_overlap_of_mask(true_mask, score_mask)
    
