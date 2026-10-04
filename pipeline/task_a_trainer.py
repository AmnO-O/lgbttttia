import os
import time
import json
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .config import PipelineConfig, IDX2STEREOTYPE
from .models.mmbert import unfreeze_last_n
from .losses import FocalLoss

class TaskATrainer:
    """
    Dedicated Trainer for Task A: Class-Aware Multi-Head Cross-Attention (MHCA) Architecture.
    Trains stereotype classification (2 classes: no, yes) with:
      - Explicit Role Embeddings (<T>, <D>, <C>)
      - Learned Class Queries [q_NonStereotype, q_Stereotype]
      - Optional Query Interaction (MHSA) Ablation
      - Consecutive Cross-Attention Query Refinement Stack (default 3 hops)
      - Exact example-weighted loss averaging
      - Guaranteed sample index tracking & validation alignment
      - 2-Phase Fine-Tuning (Frozen backbone -> unfreeze last N layers)
    """
    def __init__(
        self,
        model: nn.Module,
        config: PipelineConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        df_val: pd.DataFrame,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.df_val = df_val

        # Device assignment
        if config.device:
            self.device = torch.device(config.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        
        # Mixed Precision (AMP) support for GPU speedup
        self.use_amp = (self.device.type == "cuda")
        device_type = self.device.type if self.device.type in ["cuda", "cpu"] else "cuda"
        self.device_type = device_type
        
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(device_type, enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            
        if self.use_amp:
            print(f"[TaskATrainer] PyTorch Automatic Mixed Precision (AMP - FP16 on {self.device}) Enabled.")
        
        # Configure Loss Function: Focal Loss or CrossEntropyLoss with balanced weights & label smoothing
        class_weights = getattr(config, 'class_weights', None)
        label_smoothing = getattr(config, 'label_smoothing', 0.05)
        loss_type = getattr(config, 'loss_type', 'focal')
        focal_gamma = getattr(config, 'focal_gamma', 2.0)
        
        weights_tensor = None
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        if loss_type == "focal":
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                alpha=weights_tensor,
                label_smoothing=label_smoothing,
                reduction="mean"
            )
            print(f"[TaskATrainer] Loss: Multi-Class Focal Loss (gamma={focal_gamma}, alpha={class_weights}, label_smoothing={label_smoothing})")
        else:
            if weights_tensor is not None:
                self.criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
                print(f"[TaskATrainer] Loss: Weighted CrossEntropyLoss (weights={class_weights}, label_smoothing={label_smoothing})")
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                print(f"[TaskATrainer] Loss: Standard CrossEntropyLoss (label_smoothing={label_smoothing})")
            
        self.history = []
        os.makedirs(self.config.output_dir, exist_ok=True)

    def build_optimizer(self, lr: float, backbone_lr: Optional[float] = None) -> torch.optim.Optimizer:
        backbone_prms = [p for n, p in self.model.named_parameters()
                         if p.requires_grad and n.startswith('mmbert.')]
        head_prms = [p for n, p in self.model.named_parameters()
                     if p.requires_grad and not n.startswith('mmbert.')]

        if backbone_prms and backbone_lr is not None:
            return torch.optim.AdamW([
                {'params': backbone_prms, 'lr': backbone_lr, 'weight_decay': self.config.weight_decay},
                {'params': head_prms, 'lr': lr, 'weight_decay': self.config.weight_decay},
            ])
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=self.config.weight_decay
        )

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in self.train_loader:
            input_ids, attention_mask, role_ids, sample_indices, st_labels, hs_labels, tg_labels = batch
            batch_size = st_labels.size(0)
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            role_ids = role_ids.to(self.device, non_blocking=True)
            st_labels = st_labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                logits, _, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    role_ids=role_ids
                )
                loss = self.criterion(logits, st_labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad_norm)
                optimizer.step()

            total_loss += loss.item() * batch_size
            total_examples += batch_size

        return total_loss / max(total_examples, 1)

    @torch.no_grad()
    def eval_epoch(self) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        all_indices = []
        all_preds = []
        all_probs = []
        all_labels = []

        for batch in self.val_loader:
            input_ids, attention_mask, role_ids, sample_indices, st_labels, hs_labels, tg_labels = batch
            batch_size = st_labels.size(0)
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            role_ids = role_ids.to(self.device, non_blocking=True)
            st_labels = st_labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                logits, _, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    role_ids=role_ids
                )
                loss = self.criterion(logits, st_labels)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_indices.append(sample_indices.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(st_labels.cpu().numpy())

        avg_loss = total_loss / max(total_examples, 1)
        indices = np.concatenate(all_indices)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs, axis=0)

        # Sort / reorder by original df_val row indices to guarantee 100% alignment
        if len(indices) == len(self.df_val):
            sort_order = np.argsort(indices)
            indices = indices[sort_order]
            y_pred = y_pred[sort_order]
            y_true = y_true[sort_order]
            y_prob = y_prob[sort_order]

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0))
        f1_per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
        
        metrics = {
            'st_acc': acc,
            'st_macro_f1': macro_f1,
            'st_f1_no': float(f1_per_class[0]),
            'st_f1_yes': float(f1_per_class[1]),
        }
        return avg_loss, metrics, y_pred, y_prob, indices

    def run_training_loop(self, tag: str, num_epochs: int, lr: float,
                          backbone_lr: Optional[float] = None) -> str:
        optimizer = self.build_optimizer(lr=lr, backbone_lr=backbone_lr)
        best_metric = -1.0
        best_val_loss = float('inf')
        counter = 0
        best_checkpoint_path = os.path.join(self.config.output_dir, f"{tag}.pt")

        print(f"\n--- [Task A Class-Aware] Starting {tag} (Max Epochs: {num_epochs}, Patience: {self.config.patience}, Head LR: {lr}) ---")

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(optimizer)
            val_loss, metrics, _, _, _ = self.eval_epoch()
            elapsed = time.time() - t0

            epoch_record = {
                'epoch': epoch + 1,
                'tag': tag,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'time_s': elapsed,
                **metrics
            }
            self.history.append(epoch_record)

            print(
                f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Macro F1: {metrics['st_macro_f1']:.4f} (No: {metrics['st_f1_no']:.3f}, Yes: {metrics['st_f1_yes']:.3f}) | {elapsed:.1f}s"
            )

            current_f1 = metrics['st_macro_f1']
            is_better = (current_f1 > best_metric + 1e-4) or (abs(current_f1 - best_metric) <= 1e-4 and val_loss < best_val_loss)
            
            if is_better:
                best_metric = current_f1
                best_val_loss = val_loss
                counter = 0
                torch.save(self.model.state_dict(), best_checkpoint_path)
                print(f"  --> Saved Best Checkpoint (Macro-F1: {current_f1:.4f}): {best_checkpoint_path}")
            else:
                counter += 1
                if counter >= self.config.patience:
                    print(f"  --> Early stopping triggered at epoch {epoch+1} (patience reached).")
                    break

        return best_checkpoint_path

    def train(self) -> Dict[str, Any]:
        print(f"[Task A Class-Aware Attention Model] Device: {self.device}")
        print(f"Query Interaction Layer (MHSA) Enabled: {self.model.use_query_interaction}")

        if self.config.two_phase:
            if self.config.freeze_phase_epochs > 0:
                # PHASE 1: Frozen Backbone
                print("\n>>> PHASE 1/2: Frozen mmBERT Backbone -> Learning Role Embeddings, Stereotype Class Queries & MHCA")
                for p in self.model.mmbert.parameters():
                    p.requires_grad = False

                phase1_tag = "task_a_phase1_frozen"
                best_p1 = self.run_training_loop(
                    tag=phase1_tag,
                    num_epochs=self.config.freeze_phase_epochs,
                    lr=self.config.learning_rate
                )

                # PHASE 2: Load Phase 1, unfreeze last N layers
                print(f"\n>>> Loading Best Phase 1 Weights: {best_p1}")
                self.model.load_state_dict(torch.load(best_p1, map_location=self.device))
            else:
                print("\n>>> Skipping Phase 1 (freeze_phase_epochs = 0)")

            n_unfrozen = unfreeze_last_n(self.model.mmbert, self.config.unfreeze_layers)
            print(f">>> PHASE 2/2: Fine-Tuning Last {self.config.unfreeze_layers} Encoder Blocks (Found {n_unfrozen})")

            phase2_tag = "task_a_best_model"
            best_final = self.run_training_loop(
                tag=phase2_tag,
                num_epochs=self.config.unfreeze_phase_epochs,
                lr=self.config.head_unfreeze_lr,
                backbone_lr=self.config.unfreeze_lr
            )
        else:
            best_final = self.run_training_loop(
                tag="task_a_best_model",
                num_epochs=self.config.epochs,
                lr=self.config.learning_rate
            )

        # Load best weights
        self.model.load_state_dict(torch.load(best_final, map_location=self.device))
        _, final_metrics, y_pred, y_prob, indices = self.eval_epoch()

        print("\n==================== FINAL TASK A METRICS ====================")
        print(f"  Overall Accuracy:  {final_metrics['st_acc']:.4f}")
        print(f"  Macro-F1:          {final_metrics['st_macro_f1']:.4f}")
        print(f"  F1 (No Stereotype): {final_metrics['st_f1_no']:.4f}")
        print(f"  F1 (Stereotype):    {final_metrics['st_f1_yes']:.4f}")
        print("==============================================================\n")

        # Save predictions CSV safely reconstructed with indices
        if self.config.save_predictions and 'StereoQueerEval_id' in self.df_val.columns:
            aligned_df = self.df_val.iloc[indices].copy() if len(indices) == len(self.df_val) else self.df_val.copy()
            res_df = aligned_df[['StereoQueerEval_id', 'lang', 'yt_title', 'yt_comment', 'stereotype']].copy()
            res_df['pred_stereotype'] = [IDX2STEREOTYPE.get(int(p), 'no') for p in y_pred]
            res_df['prob_no'] = y_prob[:, 0]
            res_df['prob_yes'] = y_prob[:, 1]
            pred_file = os.path.join(self.config.output_dir, "task_a_val_predictions.csv")
            res_df.to_csv(pred_file, index=False)
            print(f"Predictions saved to: {pred_file}")

        # Save history
        hist_path = os.path.join(self.config.output_dir, "task_a_training_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            'checkpoint_path': best_final,
            'final_metrics': final_metrics,
            'history': self.history
        }
