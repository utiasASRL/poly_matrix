import itertools
import sys
from os.path import dirname

import numpy as np
import pytest

from poly_matrix import PolyMatrix

sys.path.append(dirname(__file__) + "/../")
print("appended:", sys.path[-1])


def get_simple_poly():
    A_poly1 = PolyMatrix()
    var_dict = {"x1": 2, "x2": 2, "z1": 1, "z2": 1}
    B = np.arange(4).reshape((2, 2))
    A_poly1["x1", "x1"] = 0.5 * (B + B.T)
    A_poly1["x2", "x2"] = B + B.T
    A_poly1["z1", "x1"] = np.full((1, 2), 1.0)
    A_poly1["z2", "x2"] = np.full((1, 2), 2.0)
    return A_poly1, var_dict


def test_init_from_sparse():
    A_poly1, var_dict = get_simple_poly()
    A_sparse1 = A_poly1.get_matrix(var_dict)

    A_poly2 = PolyMatrix()
    A_poly2.init_from_sparse(A_sparse1, var_dict)
    A_sparse2 = A_poly2.get_matrix(var_dict)
    np.testing.assert_allclose(A_sparse1.toarray(), A_sparse2.toarray())


def test_interpret():
    A_poly, var_dict = get_simple_poly()
    df = A_poly.interpret(var_dict)
    for i, val in df.items():
        keyi, keyj = i.split(".")
        if not np.any(np.isnan(val)):
            np.testing.assert_allclose(val, A_poly[keyi, keyj])


if __name__ == "__main__":
    test_init_from_sparse()
    test_interpret()
