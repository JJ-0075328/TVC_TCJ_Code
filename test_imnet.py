"""
Unit Tests for IMNeT Architecture and Preprocessing.
Tests:
1. Audio Processor & Acoustic Feature Extractor
2. Video Processor & Landmark Motion Extractor
3. MIA-GRU Cell & Video Encoder
4. Audio 1D-CNN + GRU Encoder
5. Siamese Identity Consistency Encoder
6. Multimodal Fusion & 4-Head Classifier
7. End-to-End Forward & Loss Backward
8. Single-file training.py standalone import & verification
"""

import unittest
import numpy as np
import torch

from training import (
    IMNeTConfig,
    IMNeT,
    AudioPreprocessor,
    VideoPreprocessor,
    AudioEncoder,
    MIAGRUVideoEncoder,
    SiameseIdentityConsistencyEncoder,
    MultiBranchClassifier,
    IMNeTHybridLoss,
    IdentityAwareAdam
)

class TestIMNeT(unittest.TestCase):
    def setUp(self):
        self.config = IMNeTConfig()

    def test_audio_preprocessing_and_acoustic_features(self):
        audio = np.random.randn(32000).astype(np.float32) # 2 seconds at 16kHz
        proc = AudioPreprocessor(target_sr=16000)
        norm_audio = proc.normalize_zscore(audio)
        self.assertEqual(len(norm_audio.shape), 1)

        features = proc.extract_acoustic_features(norm_audio)
        self.assertEqual(features.shape[-1], 15)  # 13 MFCC + 1 Pitch + 1 Energy
        self.assertGreater(features.shape[0], 0)

    def test_video_preprocessing_and_landmarks(self):
        proc = VideoPreprocessor(target_size=(256, 256), num_landmarks=68)
        self.assertEqual(proc.canonical_lm.shape, (68, 2))

    def test_audio_encoder(self):
        encoder = AudioEncoder(in_dim=15, hidden_dim=128, embed_dim=128)
        dummy_in = torch.randn(4, 100, 15) # (B, T, 15)
        out = encoder(dummy_in)
        self.assertEqual(out.shape, (4, 128))

    def test_mia_gru_video_encoder(self):
        encoder = MIAGRUVideoEncoder(num_landmarks=68, hidden_dim=128, embed_dim=128, alpha=0.5)
        dummy_lm = torch.randn(4, 30, 68, 2) # (B, T, N, 2)
        out = encoder(dummy_lm)
        self.assertEqual(out.shape, (4, 128))

    def test_siamese_encoder(self):
        siamese = SiameseIdentityConsistencyEncoder(embed_dim=128, margin=1.0)
        x1 = torch.randn(4, 3, 256, 256)
        x2 = torch.randn(4, 3, 256, 256)
        e1, e2, dist = siamese(x1, x2)
        self.assertEqual(e1.shape, (4, 128))
        self.assertEqual(e2.shape, (4, 128))
        self.assertEqual(dist.shape, (4,))

    def test_multibranch_classifier(self):
        classifier = MultiBranchClassifier(audio_dim=128, video_dim=128, id_dim=128, fusion_dim=128, num_heads=4)
        ea = torch.randn(4, 128)
        ev = torch.randn(4, 128)
        eid = torch.randn(4, 128)
        probs, fused = classifier(ea, ev, eid)
        self.assertEqual(probs.shape, (4, 4))
        self.assertEqual(fused.shape, (4, 128))

    def test_end_to_end_model_and_loss(self):
        model = IMNeT(self.config)
        criterion = IMNeTHybridLoss(lambda_ic=0.5, margin=1.0)
        optimizer = IdentityAwareAdam(model.parameters(), lr=0.0001)

        af = torch.randn(2, 50, 15)
        lm = torch.randn(2, 20, 68, 2)
        ff = torch.randn(2, 3, 256, 256)
        pair_ff = torch.randn(2, 3, 256, 256)
        y_id = torch.tensor([1.0, 0.0])
        target_labels = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])

        outputs = model(af, lm, ff, pair_face_frames=pair_ff)
        loss, cls_loss, ic_loss = criterion(
            pred_probs=outputs["probabilities"],
            target_labels=target_labels,
            dist=outputs["id_distance"],
            y_id=y_id
        )

        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        optimizer.set_identity_weight(1.2)
        optimizer.step()

        # Check gradients
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
        self.assertTrue(has_grad)

if __name__ == "__main__":
    unittest.main()
