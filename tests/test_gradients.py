import pytest

torch = pytest.importorskip("torch")
from ecom.smoke import dpo_loss


def test_dpo_gradient_matches_finite_difference_and_direction():
    chosen = torch.tensor(-3.0, requires_grad=True)
    rejected = torch.tensor(-4.0, requires_grad=True)
    rc = torch.tensor(-3.0)
    rr = torch.tensor(-4.0)
    loss = dpo_loss(chosen, rejected, rc, rr, beta=0.2)
    loss.backward()
    assert chosen.grad < 0 and rejected.grad > 0
    eps = 1e-3
    numeric = (
        dpo_loss(chosen.detach() + eps, rejected.detach(), rc, rr, 0.2)
        - dpo_loss(chosen.detach() - eps, rejected.detach(), rc, rr, 0.2)
    ) / (2 * eps)
    assert torch.allclose(chosen.grad, numeric, atol=1e-4)
