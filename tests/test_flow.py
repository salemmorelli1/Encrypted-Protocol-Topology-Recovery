import torch

from encrypted_topology.flows import MADE, MaskedAutoregressiveFlow


def test_made_first_coordinate_is_input_independent():
    torch.manual_seed(7)
    made = MADE(4, hidden=16)
    first = torch.randn(3, 4)
    second = torch.randn(3, 4)
    shift_a, scale_a = made(first)
    shift_b, scale_b = made(second)
    assert torch.allclose(shift_a[:, 0], shift_b[:, 0])
    assert torch.allclose(scale_a[:, 0], scale_b[:, 0])


def test_maf_round_trip_and_jacobian_sign():
    torch.manual_seed(11)
    flow = MaskedAutoregressiveFlow(4, hidden=16, layers=2)
    base = torch.randn(5, 4)
    transformed, forward_log_det = flow.from_base(base)
    recovered, inverse_log_det = flow.to_base(transformed)
    assert torch.allclose(base, recovered, atol=1e-5, rtol=1e-5)
    assert torch.allclose(forward_log_det, -inverse_log_det, atol=1e-5, rtol=1e-5)

