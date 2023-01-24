import numpy as np
import pytest

import sys
from os.path import dirname

sys.path.append(dirname(__file__) + "/../")

from poly_matrix import PolyMatrix


def get_Ai(test=False):
    """
    Creates the following matrices:

    A_0
        x1    x2    z1 z2 l
    x1  0  0  0  0  0  0  0
        *  0  0  0  0  0  0
    x2  *  *  0  0  0  0  0
        *  *  *  0  0  0  0
    z1  *  *  *  *  0  0  0
    z2  *  *  *  *  *  0  0
     l  *  *  *  *  *  *  1

    A_1
        x1    x2    z1 z2 l
    x1  1  0  0  0  0  0  0
        *  1  0  0  0  0  0
    x2  *  *  0  0  0  0  0
        *  *  *  0  0  0  0
    z1  *  *  *  *  0  0  -0.5
    z2  *  *  *  *  *  0  0
     l  *  *  *  *  *  *  0

    A_2
        x1    x2    z1 z2 l
    x1  0  0  0  0  0  0  0
        *  0  0  0  0  0  0
    x2  *  *  1  0  0  0  0
        *  *  *  1  0  0  0
    z1  *  *  *  *  0  0  0
    z2  *  *  *  *  *  0  -0.5
     l  *  *  *  *  *  *  0


    """

    variables = ["x1", "x2", "z1", "z2", "l"]

    A_0 = PolyMatrix()
    A_0["l", "l"] = 1.0
    A_0.print(variables=variables)

    A_list = []
    for n in range(1, 3):
        A_n = PolyMatrix()
        A_n[f"x{n}", f"x{n}"] = np.eye(2)
        A_n[f"z{n}", "l"] = -0.5
        A_n.print(variables=variables)
        A_list.append(A_n)
    return A_0, A_list


def get_Q(test=False):
    """
    Creates the following matrix:

        x1    x2    z1 z2 l
    x1  1  2  3  4  7  0  0
        *  3  4  5  8  0  0
    x2  *  *  0  0  0  9  0
        *  *  *  0  0  10 0
    z1  *  *  *  *  3  0  0
    z2  *  *  *  *  *  2  0
     l  *  *  *  *  *  *  1
    """

    Q = PolyMatrix()

    if test:
        with pytest.raises(Exception):
            # not symmetric
            Q["x1", "x1"] = np.r_[np.c_[1, 2], np.c_[3, 4]]
    Q["x1", "x1"] = np.r_[np.c_[1, 2], np.c_[2, 3]]
    if test:
        with pytest.raises(Exception):
            # inconsistent dimensions (clashes with x1-x1 block)
            Q["x1", "x2"] = np.r_[np.c_[3, 4], np.c_[4, 5], np.c_[1, 1]]
        with pytest.raises(Exception):
            Q["x1", "x1"] = np.r_[np.c_[3, 4], np.c_[4, 5], np.c_[1, 1]]
    Q["x1", "x2"] = np.r_[np.c_[3, 4], np.c_[4, 5]]
    Q["x1", "z1"] = np.c_[[7, 8]]
    Q["x2", "z2"] = np.c_[[9, 10]]
    Q["z1", "z1"] = 3

    if test:
        with pytest.raises(Exception):
            # inconsistent dimensions (clashes with x2-z2 block)
            Q["z2", "z2"] = np.r_[[1, 2]]
    Q["z2", "z2"] = 2
    Q["l", "l"] = 1
    return Q


def test_Q():
    get_Q(test=True)


def test_Ai():
    get_Ai(test=True)


def test_operations():

    # mat1 =
    # 1 0
    # 0 2
    mat1 = PolyMatrix()
    mat1[0, 0] = 1.0
    mat1[1, 1] = 2.0

    mat0 = PolyMatrix()
    mat0 += mat0 + mat1

    # mat2 =
    # 3 1
    # 1 0
    mat2 = PolyMatrix()
    mat2[0, 0] = 3.0
    mat2[0, 1] = 1.0

    mat3 = mat1 + mat2
    assert mat3[0, 0] == 4
    assert mat3[1, 1] == 2
    assert mat3[0, 1] == 1
    assert mat3[1, 0] == 1

    mat4 = mat1 - mat2
    assert mat4[0, 0] == -2
    assert mat4[1, 1] == 2
    assert mat4[0, 1] == -1
    assert mat4[1, 0] == -1

    mat5 = mat1 * 2
    assert mat5[0, 0] == 2
    assert mat5[1, 1] == 4
    assert mat5[0, 1] is None
    assert mat5[1, 0] is None

    mat6 = mat2 * 2
    assert mat6[0, 0] == 6
    assert mat6[1, 1] is None
    assert mat6[0, 1] == 2
    assert mat6[1, 0] == 2

    mat1 += mat2
    assert mat1[0, 0] == 4
    assert mat1[1, 1] == 2
    assert mat1[0, 1] == 1
    assert mat1[1, 0] == 1

    mat1 *= 2
    assert mat1[0, 0] == 8
    assert mat1[1, 1] == 4
    assert mat1[1, 0] == 2
    assert mat1[0, 1] == 2


def test_addition():
    Q = get_Q()
    A_0, A_list = get_Ai()

    H_mat = Q.get_matrix()

    H = Q.copy()
    np.testing.assert_equal(H.get_matrix().toarray(), H_mat.toarray())

    A_0.reorder(Q.variable_dict)
    H_mat += A_0.get_matrix() * 2.0
    H += A_0 * 2.0
    np.testing.assert_equal(H.get_matrix().toarray(), H_mat.toarray())

    for A_n in A_list:
        A_n.reorder(Q.variable_dict)
        H_mat += A_n.get_matrix()
        H += A_n
        np.testing.assert_equal(H.get_matrix().toarray(), H_mat.toarray())


def test_reorder():
    Q = get_Q()

    variables = list(Q.variable_dict.keys())

    variables_new = variables[::-1]
    Q_new = Q.copy()
    Q_new.reorder(variables_new)

    # assert cost before and after reordering is the same
    f_dict = {"x1": [0, 1], "x2": [2, 3], "z1": 1, "z2": 3, "l": 1}
    f_new = Q_new.get_vector(**f_dict)
    f = Q.get_vector(**f_dict)

    Q_sparse = Q.get_matrix()
    Q_new_sparse = Q_new.get_matrix()

    cost = f.T @ Q_sparse @ f
    cost_new = f_new.T @ Q_new_sparse @ f_new
    assert cost == cost_new


if __name__ == "__main__":
    test_reorder()

    test_operations()

    test_Ai()
    test_Q()

    test_addition()

    print("all tests passed")
