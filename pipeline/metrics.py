from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from .data import decode_target

@torch.no_grad()
def run_model_inference(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                        device: torch.device, is_mmbert_tf: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs a forward pass across the data loader, collecting sample indices and prediction arrays."""
    model.eval()
    idx_list, st_list, hs_list, tg_list = [], [], [], []

    for batch in loader:
        if is_mmbert_tf:
            ids, mask, sample_idx = batch[0].to(device), batch[1].to(device), batch[2]
            st_logits, hs_logits, tg_logits = model(ids, mask)
        else:
            inputs, sample_idx = batch[0].to(device), batch[1]
            st_logits, hs_logits, tg_logits = model(inputs)

        idx_list.append(sample_idx.cpu().numpy())
        st_list.append(torch.sigmoid(st_logits.squeeze(-1)).cpu().numpy())
        hs_list.append(torch.argmax(hs_logits, dim=1).cpu().numpy())
        tg_list.append(torch.sigmoid(tg_logits).cpu().numpy())

    all_indices = np.concatenate(idx_list)
    st_probs = np.concatenate(st_list).ravel()
    hs_preds = np.concatenate(hs_list)
    tg_probs = np.concatenate(tg_list, axis=0)

    # Sort / reorder by original df row indices to guarantee alignment regardless of DataLoader batch ordering
    if len(all_indices) > 0:
        sort_order = np.argsort(all_indices)
        all_indices = all_indices[sort_order]
        st_probs = st_probs[sort_order]
        hs_preds = hs_preds[sort_order]
        tg_probs = tg_probs[sort_order]

    return st_probs, hs_preds, tg_probs, all_indices


def evaluate_stereoqueer(
    model: torch.nn.Module,
    df: pd.DataFrame,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    is_mmbert_tf: bool = True
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Computes official StereoQueerEval evaluation metrics with guaranteed index alignment:
      - Stereotype: accuracy + macro-F1
      - Hate speech: accuracy + macro-F1 (3 classes: no, implicit, explicit)
      - Target: string exact-match accuracy against gold annotations
      - Per-language metrics breakdown (EN, IT, NL)
    """
    st_probs, hs_preds, tg_probs, indices = run_model_inference(model, loader, device, is_mmbert_tf)
    st_preds = (st_probs >= 0.5).astype(int)
    tg_pred_str = np.array([decode_target(v) for v in tg_probs])

    # Reconstruct prediction table using sample indices to guarantee 100% alignment
    aligned_df = df.iloc[indices].reset_index(drop=True) if len(indices) == len(df) else df

    gold_st = aligned_df['st_y'].values.astype(int)
    gold_hs = aligned_df['hs_y'].values.astype(int)
    gold_tg = aligned_df['target'].fillna('none').values

    metrics = {
        'st_acc': float(accuracy_score(gold_st, st_preds)),
        'st_f1': float(f1_score(gold_st, st_preds, labels=[0, 1], average='macro', zero_division=0)),
        'hs_acc': float(accuracy_score(gold_hs, hs_preds)),
        'hs_f1': float(f1_score(gold_hs, hs_preds, labels=[0, 1, 2], average='macro', zero_division=0)),
        'tg_exact_match': float((tg_pred_str == gold_tg).mean()),
    }
    # Weighted composite score
    metrics['macro_avg_f1'] = float((metrics['st_f1'] + metrics['hs_f1'] + metrics['tg_exact_match']) / 3.0)

    # Per-language breakdown if available
    per_language = {}
    if 'lang' in aligned_df.columns:
        for lang in sorted(aligned_df['lang'].unique()):
            idx = (aligned_df['lang'].values == lang)
            sub_gold_st = gold_st[idx]
            sub_st_preds = st_preds[idx]
            sub_gold_hs = gold_hs[idx]
            sub_hs_preds = hs_preds[idx]
            sub_gold_tg = gold_tg[idx]
            sub_tg_preds = tg_pred_str[idx]

            per_language[lang] = {
                'count': int(idx.sum()),
                'st_f1': float(f1_score(sub_gold_st, sub_st_preds, labels=[0, 1], average='macro', zero_division=0)),
                'hs_f1': float(f1_score(sub_gold_hs, sub_hs_preds, labels=[0, 1, 2], average='macro', zero_division=0)),
                'tg_exact_match': float((sub_tg_preds == sub_gold_tg).mean()),
            }

    predictions = {
        'st_probs': st_probs,
        'st_preds': st_preds,
        'hs_preds': hs_preds,
        'tg_pred_str': tg_pred_str,
        'indices': indices,
        'per_language': per_language
    }

    return metrics, predictions


def print_metrics(metrics: Dict[str, float], title: str = "VALIDATION"):
    """Pretty prints evaluation metrics to console."""
    print(f"\n==================== {title} ====================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.4f}")
        else:
            print(f"  {k:<18}: {v}")
    print("==================================================\n")
