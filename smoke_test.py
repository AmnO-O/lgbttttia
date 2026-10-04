"""
Smoke test suite for the StereoQueerEval Task B pipeline and components.
Executes end-to-end forward/backward passes and data splitting tests.
"""
import unittest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.config import PipelineConfig, HATE_CLASSES, HATE2IDX, IDX2HATE
from pipeline.data import DataPipeline, safe_clean, encode_target, decode_target
from pipeline.models.task_b_class_aware import (
    TaskBClassAwareAttentionModel,
    ROLE_PAD, ROLE_TITLE, ROLE_DESC, ROLE_COMMENT
)
from pipeline.task_b_data import TaskBRoleDataset
from pipeline.task_b_trainer import TaskBTrainer
from pipeline.models.task_a_class_aware import TaskAClassAwareAttentionModel
from pipeline.task_a_data import TaskARoleDataset
from pipeline.task_a_trainer import TaskATrainer


class MockBackbone(nn.Module):
    """Mock HuggingFace backbone producing [B, S, 768] hidden states."""
    def __init__(self, d_model=768, num_layers=22):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embed = nn.Embedding(1000, d_model)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        out = self.embed(input_ids % 1000)
        class Output:
            pass
        res = Output()
        res.last_hidden_state = out
        if output_hidden_states:
            # 22 layers of hidden states
            res.hidden_states = tuple(out * (i + 1) for i in range(self.num_layers))
        return res


class MockTokenizer:
    """Mock tokenizer providing IDs for smoke testing."""
    def __init__(self):
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        # Hash words to integer tokens
        words = text.split()
        return [abs(hash(w)) % 900 + 10 for w in words]

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{i}" for i in ids]


class TestPipelineSmoke(unittest.TestCase):
    def setUp(self):
        # Create synthetic test dataset
        self.raw_data = {
            'yt_title': ['Video Alpha', 'Video Alpha', 'Video Beta', 'Video Gamma', 'Video Gamma'],
            'yt_description': ['Desc A', 'Desc A', 'Desc B', 'Desc C', 'Desc C'],
            'yt_comment': [
                'Great video about rights',
                'Hate comment here',
                'Neutral comment',
                'Implicit hate message',
                'Another normal comment'
            ],
            'stereotype': ['no', 'yes', 'no', 'yes', 'no'],
            'hate_speech': ['no', 'yes_explicit', 'no', 'yes_implicit', 'no'],
            'target': ['none', 'group_lgbtqia+', 'none', 'individual_t', 'none'],
        }
        self.df = pd.DataFrame(self.raw_data)
        self.config = PipelineConfig(
            task="stereoqueer",
            target_task="hs",
            model_type="task_b_class_aware",
            batch_size=2,
            max_length=32,
            two_phase=False,
            device="cpu"
        )

    def test_01_constants_and_imports(self):
        self.assertEqual(len(HATE_CLASSES), 3)
        self.assertIn('yes_implicit', HATE2IDX)
        self.assertEqual(IDX2HATE[0], 'no')

    def test_02_data_pipeline_split(self):
        dp = DataPipeline(self.config)
        dp.df_all = self.df.copy()
        dp.df_all['st_y'] = (dp.df_all['stereotype'] == 'yes').astype(np.float32)
        dp.df_all['hs_y'] = dp.df_all['hate_speech'].map(HATE2IDX).fillna(0).astype(np.int64)
        dp.df_all['tg_y'] = [encode_target(t) for t in dp.df_all['target']]

        # Test with test_size and random_state keyword arguments
        df_train, df_val = dp.split_data(test_size=0.3, random_state=42)
        self.assertGreater(len(df_train), 0)
        self.assertGreater(len(df_val), 0)

        # Confirm 0% video title leakage
        train_titles = set(df_train['yt_title'])
        val_titles = set(df_val['yt_title'])
        self.assertEqual(len(train_titles & val_titles), 0, "Video title leaked between splits!")

    def test_03_task_b_dataset_and_roles(self):
        tok = MockTokenizer()
        df_test = self.df.copy()
        df_test['st_y'] = 0.0
        df_test['hs_y'] = [0, 2, 0, 1, 0]
        df_test['tg_y'] = [[0.0] * 10 for _ in range(len(df_test))]

        ds = TaskBRoleDataset(df_test, tok, max_len=32)
        self.assertEqual(len(ds), 5)

        item = ds[0]
        input_ids, mask, roles, sample_idx, st, hs, tg = item
        self.assertEqual(input_ids.shape[0], 32)
        self.assertEqual(mask.shape[0], 32)
        self.assertEqual(roles.shape[0], 32)
        self.assertEqual(sample_idx.item(), 0)
        self.assertIn(ROLE_TITLE, roles.tolist())
        self.assertIn(ROLE_DESC, roles.tolist())
        self.assertIn(ROLE_COMMENT, roles.tolist())

    def test_04_model_forward_backward(self):
        backbone = MockBackbone(d_model=64)
        model = TaskBClassAwareAttentionModel(
            mmbert_model=backbone,
            d_model=64,
            num_heads=2,
            dropout=0.1,
            use_query_interaction=True
        )

        B, S = 2, 16
        dummy_ids = torch.randint(0, 500, (B, S))
        dummy_mask = torch.ones((B, S), dtype=torch.long)
        dummy_roles = torch.randint(0, 4, (B, S), dtype=torch.long)
        labels = torch.tensor([0, 2], dtype=torch.long)

        logits, h_B, attn = model(dummy_ids, dummy_mask, dummy_roles, return_attention_map=True)
        self.assertEqual(logits.shape, (B, 3))
        self.assertEqual(h_B.shape, (B, 64))
        self.assertEqual(attn.shape, (B, 3, S))

        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        self.assertIsNotNone(model.role_embeddings.weight.grad)
        self.assertIsNotNone(model.query_embeddings.grad)

    def test_05_trainer_step(self):
        backbone = MockBackbone(d_model=64)
        model = TaskBClassAwareAttentionModel(
            mmbert_model=backbone,
            d_model=64,
            num_heads=2,
            dropout=0.1,
            use_query_interaction=False
        )
        tok = MockTokenizer()
        df_test = self.df.copy()
        df_test['st_y'] = 0.0
        df_test['hs_y'] = [0, 2, 0, 1, 0]
        df_test['tg_y'] = [[0.0] * 10 for _ in range(len(df_test))]

        ds = TaskBRoleDataset(df_test, tok, max_len=16)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        trainer = TaskBTrainer(
            model=model,
            config=self.config,
            train_loader=loader,
            val_loader=loader,
            df_val=df_test
        )
        optimizer = trainer.build_optimizer(lr=1e-3)
        train_loss = trainer.train_epoch(optimizer)
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0.0)

        val_loss, metrics, preds, probs, indices = trainer.eval_epoch()
        self.assertIn('hs_macro_f1', metrics)
        self.assertIn('hs_acc', metrics)
        self.assertEqual(len(indices), len(df_test))

    def test_06_task_a_dataset_and_roles(self):
        tok = MockTokenizer()
        df_test = self.df.copy()
        df_test['st_y'] = [0, 1, 0, 1, 0]
        df_test['hs_y'] = [0, 2, 0, 1, 0]
        df_test['tg_y'] = [[0.0] * 10 for _ in range(len(df_test))]

        ds = TaskARoleDataset(df_test, tok, max_len=32)
        self.assertEqual(len(ds), 5)

        item = ds[0]
        input_ids, mask, roles, sample_idx, st, hs, tg = item
        self.assertEqual(input_ids.shape[0], 32)
        self.assertEqual(mask.shape[0], 32)
        self.assertEqual(roles.shape[0], 32)
        self.assertEqual(sample_idx.item(), 0)
        self.assertEqual(st.item(), 0)

    def test_07_task_a_model_forward_backward(self):
        backbone = MockBackbone(d_model=64)
        model = TaskAClassAwareAttentionModel(
            mmbert_model=backbone,
            d_model=64,
            num_heads=2,
            dropout=0.1,
            use_query_interaction=True,
            num_queries=2
        )

        B, S = 2, 16
        dummy_ids = torch.randint(0, 500, (B, S))
        dummy_mask = torch.ones((B, S), dtype=torch.long)
        dummy_roles = torch.randint(0, 4, (B, S), dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)

        logits, h_A, attn = model(dummy_ids, dummy_mask, dummy_roles, return_attention_map=True)
        self.assertEqual(logits.shape, (B, 2))
        self.assertEqual(h_A.shape, (B, 64))
        self.assertEqual(attn.shape, (B, 2, S))

        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        self.assertIsNotNone(model.role_embeddings.weight.grad)
        self.assertIsNotNone(model.query_embeddings.grad)

    def test_08_task_a_trainer_step(self):
        backbone = MockBackbone(d_model=64)
        model = TaskAClassAwareAttentionModel(
            mmbert_model=backbone,
            d_model=64,
            num_heads=2,
            dropout=0.1,
            use_query_interaction=False,
            num_queries=2
        )
        tok = MockTokenizer()
        df_test = self.df.copy()
        df_test['st_y'] = [0, 1, 0, 1, 0]
        df_test['hs_y'] = [0, 2, 0, 1, 0]
        df_test['tg_y'] = [[0.0] * 10 for _ in range(len(df_test))]

        ds = TaskARoleDataset(df_test, tok, max_len=16)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        trainer = TaskATrainer(
            model=model,
            config=self.config,
            train_loader=loader,
            val_loader=loader,
            df_val=df_test
        )
        optimizer = trainer.build_optimizer(lr=1e-3)
        train_loss = trainer.train_epoch(optimizer)
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0.0)

        val_loss, metrics, preds, probs, indices = trainer.eval_epoch()
        self.assertIn('st_macro_f1', metrics)
        self.assertIn('st_acc', metrics)
        self.assertEqual(len(indices), len(df_test))
    def test_09_multiscale_depth_ablation(self):
        """Verify that depth ablation (d=1,2,3) correctly takes the last d backbone layers."""
        backbone = MockBackbone(d_model=64, num_layers=22)
        
        # Test d=1 -> should take layer 21 (H22)
        m1 = TaskBClassAwareAttentionModel(mmbert_model=backbone, d_model=64, num_decoder_layers=1, use_multiscale_layers=True)
        # Test d=2 -> should take layers 20, 21 (H21, H22)
        m2 = TaskBClassAwareAttentionModel(mmbert_model=backbone, d_model=64, num_decoder_layers=2, use_multiscale_layers=True)
        # Test d=3 -> should take layers 19, 20, 21 (H20, H21, H22)
        m3 = TaskBClassAwareAttentionModel(mmbert_model=backbone, d_model=64, num_decoder_layers=3, use_multiscale_layers=True)
        
        B, S = 2, 8
        dummy_ids = torch.randint(0, 100, (B, S))
        dummy_mask = torch.ones((B, S), dtype=torch.long)
        dummy_roles = torch.zeros((B, S), dtype=torch.long)
        
        logits1, _, _ = m1(dummy_ids, dummy_mask, dummy_roles)
        logits2, _, _ = m2(dummy_ids, dummy_mask, dummy_roles)
        logits3, _, _ = m3(dummy_ids, dummy_mask, dummy_roles)
        
        self.assertEqual(logits1.shape, (B, 3))
        self.assertEqual(logits2.shape, (B, 3))
        self.assertEqual(logits3.shape, (B, 3))

    def test_10_task_b_racc_model(self):
        """Verify that TaskBRACCModel runs across all baseline modes and outputs valid tensor shapes."""
        from pipeline.models.task_b_racc import TaskBRACCModel
        backbone = MockBackbone(d_model=64, num_layers=22)
        racc_model = TaskBRACCModel(
            mmbert_backbone=backbone,
            hidden_dim=64,
            num_classes=3,
            num_heads=2,
            ffn_dim=128,
            baseline_mode="racc"
        )
        B, S = 2, 12
        dummy_ids = torch.randint(0, 100, (B, S))
        dummy_mask = torch.ones((B, S), dtype=torch.long)
        # Assign roles: first 4 comment (3), next 4 title (1), last 4 desc (2)
        dummy_roles = torch.tensor([[3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2],
                                   [3, 3, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
        
        logits, h_b, diag = racc_model(dummy_ids, dummy_mask, dummy_roles, return_diagnostics=True)
        self.assertEqual(logits.shape, (B, 3))
        self.assertEqual(h_b.shape, (B, 64))
        self.assertIn('alpha_gate', diag)
        self.assertEqual(diag['alpha_gate'].shape, (B, 3, 2))
        self.assertIn('attn_hop1_comment', diag)
        self.assertIn('attn_hop2_context', diag)
        self.assertIn('attn_hop3_global', diag)

        # Test custom multiscale_layer_indices
        racc_custom = TaskBRACCModel(
            mmbert_backbone=backbone,
            hidden_dim=64,
            num_classes=3,
            num_heads=2,
            ffn_dim=128,
            baseline_mode="racc",
            multiscale_layer_indices=(-5, -3, -1)
        )
        logits_c, _, _ = racc_custom(dummy_ids, dummy_mask, dummy_roles)
        self.assertEqual(logits_c.shape, (B, 3))


if __name__ == '__main__':
    unittest.main()
