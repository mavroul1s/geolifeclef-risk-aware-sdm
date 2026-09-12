import math

import pytest
import torch

from geolifeclef.po_expert import POExpert, masked_po_loss


def test_full_species_model_and_masked_loss_backward():
    torch.set_num_threads(2)
    model = POExpert(num_labels=5016, input_dim=74)
    logits = model(torch.randn(3, 74))
    assert logits.shape == (3, 5016)
    targets = torch.zeros_like(logits)
    targets[0, [0, 5015]] = 1
    targets[1, 123] = 1
    targets[2, [22, 777, 2900]] = 1
    loss = masked_po_loss(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.classifier.weight.grad[4000].abs().sum() > 0


def test_per_pseudo_survey_loss_is_batch_replication_invariant():
    logits = torch.tensor([[1.0, -0.2, -0.7], [0.3, 2.0, -1.0]])
    targets = torch.tensor([[1, 0, 0], [0, 1, 1]])
    expected = masked_po_loss(logits, targets)
    repeated = masked_po_loss(logits.repeat_interleave(5, dim=0), targets.repeat_interleave(5, dim=0))
    assert repeated.item() == pytest.approx(expected.item())
    individual = torch.stack([masked_po_loss(x[None], y[None]) for x, y in zip(logits, targets)])
    assert individual.mean().item() == pytest.approx(expected.item())


def test_positive_and_background_terms_do_not_grow_with_species_counts():
    logits = torch.zeros(2, 5)
    targets = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 1, 1, 0]])
    for row in range(2):
        assert masked_po_loss(logits[row:row + 1], targets[row:row + 1]).item() == pytest.approx(1.05 * math.log(2))


def test_unobserved_species_have_explicitly_downweighted_gradients():
    logits = torch.zeros(1, 10, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0] = 1
    masked_po_loss(logits, targets).backward()
    positive = logits.grad[0, 0]
    unobserved = logits.grad[0, 1:]
    assert positive < 0
    assert (unobserved > 0).all()
    assert unobserved.sum().item() == pytest.approx(-positive.item() * 0.05)
    assert unobserved.max().item() < -positive.item() * 0.05


def test_zero_background_weight_masks_unobserved_species():
    logits = torch.zeros(1, 3, requires_grad=True)
    masked_po_loss(logits, torch.tensor([[1, 0, 0]]), unobserved_weight=0).backward()
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1:].tolist() == [0, 0]


def test_all_positive_row_and_extreme_half_logits_are_finite():
    logits = torch.tensor([[1000.0, -1000.0]], dtype=torch.float16, requires_grad=True)
    loss = masked_po_loss(logits, torch.ones_like(logits))
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(500.0)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("targets", [torch.zeros(2, 3), torch.tensor([[1, 0, 0], [0, 0, 0]])])
def test_no_positive_rows_fail_instead_of_training_false_absences(targets):
    with pytest.raises(ValueError, match="at least one positive"):
        masked_po_loss(torch.zeros_like(targets, dtype=torch.float32), targets)


@pytest.mark.parametrize("targets", [torch.tensor([[1.0, 0.5]]), torch.tensor([[2.0, 0.0]]), torch.tensor([[1.0, float("nan")]])])
def test_nonbinary_occurrence_counts_and_nonfinite_targets_are_rejected(targets):
    with pytest.raises(ValueError, match="finite binary"):
        masked_po_loss(torch.zeros_like(targets), targets)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_unobserved_weight_fails(weight):
    with pytest.raises(ValueError, match="unobserved_weight"):
        masked_po_loss(torch.zeros(1, 2), torch.tensor([[1, 0]]), weight)


def test_empty_mismatched_and_nonfinite_logits_fail():
    with pytest.raises(ValueError, match="nonempty"):
        masked_po_loss(torch.zeros(0, 2), torch.zeros(0, 2))
    with pytest.raises(ValueError, match="matching"):
        masked_po_loss(torch.zeros(1, 2), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="finite"):
        masked_po_loss(torch.tensor([[float("inf"), 0]]), torch.tensor([[1, 0]]))


def test_model_rejects_invalid_dimensions_and_input_shape():
    with pytest.raises(ValueError, match="positive integers"):
        POExpert(5016, 0)
    with pytest.raises(ValueError, match="feature matrix"):
        POExpert(5016, 3)(torch.randn(2, 4))
