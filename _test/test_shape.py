import numpy as np

from poly_matrix.poly_matrix import PolyMatrix


def test_shape():
    """
        x1  x2
        --- -----
        1 1 2 2 2  | x1
    A = 1 1 2 2 2  |
        2 2        | x2
        2 2        |
        2 2        |

        1   | x1
    B = 1   |
        3   | x2
        3   |
        3   |

    """
    A = PolyMatrix(symmetric=True)

    A["x1", "x1"] = np.ones((2, 2))
    assert A.shape == (2, 2)

    A["x1", "x2"] = 2 * np.ones((2, 3))
    assert A.shape == (5, 5)

    B = PolyMatrix(symmetric=False)
    B["x1", "h"] = np.ones((2, 1))
    assert B.shape == (2, 1)

    B["x2", "h"] = np.ones((3, 1))
    assert B.shape == (5, 1)

    C = A.multiply(B)
    assert C.shape == (5, 1)


if __name__ == "__main__":
    test_shape()
    print("all tests passed")
