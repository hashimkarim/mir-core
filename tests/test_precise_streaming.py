"""Validate the portable math independently of the deployment graph."""
import numpy as np
import torch

from mir_core.native.precise_streaming import PortableMath, ordered_sum, ordered_linear


def test_portable_nonlinearities_retain_double_accuracy():
    math = PortableMath()
    values = torch.cat((torch.linspace(-100, 100, 4001, dtype=torch.float64),
                        torch.tensor([-1e-15, -1e-9, 0, 1e-9, 1e-15], dtype=torch.float64)))
    torch.testing.assert_close(math.sigmoid(values), torch.sigmoid(values), rtol=0, atol=2e-15)
    torch.testing.assert_close(math.tanh(values), torch.tanh(values), rtol=0, atol=2e-15)
    middle = values[values.abs() <= 80]
    torch.testing.assert_close(math.exp(middle), torch.exp(middle), rtol=8e-16, atol=1e-30)


def test_affine_retains_accuracy_with_a_fixed_reduction_order():
    # The ordered pairwise tree gives zero; a sequential left sum gives one.
    assert ordered_sum(torch.tensor([1e16, 1, -1e16, 1], dtype=torch.float64)) == 0
    generator = torch.Generator().manual_seed(443)
    x = torch.randn(2, 31, generator=generator, dtype=torch.float64)
    w = torch.randn(7, 31, generator=generator, dtype=torch.float64)
    b = torch.randn(7, generator=generator, dtype=torch.float64)
    expected = x.numpy().astype(np.longdouble) @ w.numpy().astype(np.longdouble).T + b.numpy().astype(np.longdouble)
    np.testing.assert_allclose(ordered_linear(x, w, b).numpy(), expected, rtol=3e-15, atol=3e-15)
