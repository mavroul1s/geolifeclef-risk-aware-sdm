import numpy as np

from geolifeclef.data import CanonicalNPZDataset, inspect_npz


def test_canonical_loader(tmp_path):
    path = tmp_path / "split.npz"
    np.savez(path, labels=np.array([[1, 0], [0, 1]], dtype=np.float32), landsat=np.zeros((2, 4, 3), dtype=np.float32))
    summary = inspect_npz(path); dataset = CanonicalNPZDataset(path, "landsat")
    assert (summary.samples, summary.species) == (2, 2)
    assert dataset[0]["landsat"].shape == (4, 3)

