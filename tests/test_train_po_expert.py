import time

import numpy as np
from scipy import sparse
import torch

from scripts.train_po_expert import fit_expert, expert_predict


def test_po_and_zero_po_training_checkpoint_inference(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(21)
    features = rng.normal(size=(24, 8)).astype(np.float32)
    labels = np.eye(60, dtype=np.uint8)[np.arange(24)]
    po_features = rng.normal(size=(8, 8)).astype(np.float32)
    po_labels = sparse.csr_matrix(np.eye(60, dtype=np.uint8)[:8])
    for use_po in (True, False):
        name = 'po' if use_po else 'zero'
        model, report = fit_expert(features, labels, np.arange(16), np.arange(16, 20),
                                  po_features, po_labels, np.ones(8), torch.device('cpu'), tmp_path,
                                  name, time.monotonic() + 300, use_po, pretrain_epochs=1,
                                  epochs=1, minimum_epochs=1, batch_size=4)
        assert report['po_epochs'] == int(use_po)
        assert report['pa_epochs'] == 1
        assert (tmp_path / f'{name}_best.pt').is_file()
        probability = expert_predict(model, features[20:], torch.device('cpu'), time.monotonic() + 60)
        assert probability.shape == (4, 60)
        assert np.isfinite(probability).all()
