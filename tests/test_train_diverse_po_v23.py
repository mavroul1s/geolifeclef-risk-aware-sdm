import time

import numpy as np
from scipy import sparse
import torch

from scripts.train_diverse_po_v23 import fit_adaptation, predict, pretrain


def test_v23_fixed_checkpoint_training_and_deeper_model_smoke(tmp_path):
    rng = np.random.default_rng(23)
    features = rng.normal(size=(40, 6)).astype(np.float32)
    labels = np.zeros((40, 24), dtype=np.uint8)
    labels[np.arange(40), np.arange(40) % 24] = 1
    po_labels = sparse.csr_matrix(labels)
    device = torch.device("cpu")
    deadline = time.monotonic() + 300
    pretrained, pretraining = pretrain(features, po_labels, np.ones(40), device, tmp_path,
        deadline, seed=23, width=8, blocks=3, epochs=1, draws=24)
    assert pretraining["blocks"] == 3
    model, history = fit_adaptation(pretrained, "single_head", features, labels,
        np.arange(24), np.arange(24, 32), np.full((8, 24), 0.5, dtype=np.float32), [],
        features, po_labels, np.ones(40), device, tmp_path, deadline,
        seed=23, width=8, blocks=3, epochs=2, checkpoint_epochs=(1, 2))
    assert history["fixed_checkpoint_epochs"] == [1, 2]
    assert history["early_stopping_used"] is False
    assert predict(model, features[:3], device, deadline).shape == (3, 24)
