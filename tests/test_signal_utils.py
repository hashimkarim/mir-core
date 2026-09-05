from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mir_core.utils.signal import smooth_signal


@pytest.mark.parametrize("shape", [(31,), (31, 3)])
@pytest.mark.parametrize("kernel", [0, 2, 7, np.asarray([0.2, 0.6, 0.2])])
def test_smooth_signal_matches_madmom(shape: tuple[int, ...], kernel: object) -> None:
    madmom = pytest.importorskip("madmom")
    values = np.random.default_rng(42).normal(size=shape)

    expected = madmom.audio.signal.smooth(values, kernel)
    actual = smooth_signal(values, kernel)

    np.testing.assert_array_equal(actual, expected)


def test_smooth_signal_retains_madmom_validation() -> None:
    values = np.arange(8, dtype=np.float64)
    with pytest.raises(ValueError, match="kernel of size"):
        smooth_signal(values, 1)
    with pytest.raises(ValueError, match="can't smooth signal"):
        smooth_signal(values, [1.0])
    with pytest.raises(ValueError, match="either 1D or 2D"):
        smooth_signal(values.reshape(2, 2, 2), 3)


def test_lightweight_core_imports_do_not_import_madmom() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mir_core; "
                "import mir_core.preprocessing.utils; "
                "import mir_core.postprocessing.peak_picking; "
                "assert 'madmom' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
