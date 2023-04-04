import itertools
import sys
from os.path import dirname

import numpy as np
import pytest

sys.path.append(dirname(__file__) + "/../")
print("appended:", sys.path[-1])

from poly_matrix.cost_matrix import CostMatrix


def test_range_only():
    """
        For example, we want to create the following cost:

        $f(x) = sum_{i,j \in \mathcal{E}} (r_{ij} - ||t_j - a_i||)^2$

        We first rewrite each residual as $b_{ij}.T @ x$ where x contains all unknowns and $b_{ij}$ is the coefficient vector. 

    """
    N = 10
    K = 5
    D = 2

    mat_noiseless = CostMatrix()
    noise = 1e-3
    mat_noisy = CostMatrix()

    np.random.seed(1)
    anchors = np.random.rand(K, D)
    points = np.random.rand(N, D)

    edges = list(itertools.product(range(N), range(K), repeat=1))
    for n, k in edges:
        distance_sq = np.linalg.norm(anchors[k] - points[n])**2
        distance_sq_noisy = distance_sq + np.random.normal(loc=0, scale=noise)
        mat_noiseless.add_from_residual({"l": distance_sq, f"tau:{n}":-1, f"alpha:{k}":-1, f"e:{n}{k}":2})
        mat_noisy.add_from_residual({"l": distance_sq_noisy, f"tau:{n}":-1, f"alpha:{k}":-1, f"e:{n}{k}":2})

    # below may be provided by custom lifter classes.
    var_dict = {
        "l": 1,
    }
    var_dict.update({f"t:{n}": D for n in range(N)})
    var_dict.update({f"a:{k}": D for k in range(K)})
    var_dict.update({f"tau:{n}": 1 for n in range(N)})
    var_dict.update({f"alpha:{k}": 1 for k in range(K)})
    var_dict.update({f"e:{n}{k}": 1 for n, k in edges})
    x_data = [[1]]
    x_data += [p for p in points]
    x_data += [a for a in anchors]
    x_data += [[np.linalg.norm(p)**2 ]for p in points]
    x_data += [[np.linalg.norm(a)**2 ]for a in anchors]
    x_data += [[anchors[k] @ points[n]] for n, k in edges]

    x = np.concatenate(x_data)
    Q = mat_noiseless.toarray(variables=var_dict)
    err = x.T @ Q @ x
    assert abs(err) < 1e-10, err

    Q = mat_noisy.toarray(variables=var_dict)
    err = x.T @ Q @ x
    assert abs(err) < noise, err

if __name__ == "__main__":
    test_range_only()
    print("all tests passed")