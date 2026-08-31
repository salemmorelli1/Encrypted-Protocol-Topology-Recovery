import numpy as np

from encrypted_topology.experiment import ARCHITECTURES, factorial_contrast_matrix
from encrypted_topology.simulator import OBFUSCATION_LEVELS, SPARSITY_LEVELS


def test_factorial_contrast_design_is_full_rank_and_architecture_signed():
    rows = [
        {
            "seed": str(seed),
            "architecture": architecture,
            "obfuscation": obfuscation,
            "sparsity": sparsity,
        }
        for seed in (2026, 2027)
        for architecture in ARCHITECTURES
        for obfuscation in OBFUSCATION_LEVELS
        for sparsity in SPARSITY_LEVELS
    ]
    fixed, names, random, groups = factorial_contrast_matrix(rows)
    assert fixed.shape == (36, 18)
    assert np.linalg.matrix_rank(fixed) == 18
    assert random.shape == (36, 2)
    assert np.unique(groups).tolist() == [2026, 2027]
    architecture_column = fixed[:, names.index("architecture")]
    assert set(np.unique(architecture_column)) == {-0.5, 0.5}
