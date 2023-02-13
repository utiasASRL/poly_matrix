import numpy as np
import pytest

import sys
from os.path import dirname

sys.path.append(dirname(__file__) + "/../")
print("appended:", sys.path[-1])

from poly_matrix import PolyMatrix, sorted_dict


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

    # variables = ["x1", "x2", "z1", "z2", "l"]
    A_0 = PolyMatrix()
    A_0["l", "l"] = 1.0
    # A_0.print(variables=variables)

    A_list = []
    for n in range(1, 3):
        A_n = PolyMatrix()
        A_n[f"x{n}", f"x{n}"] = np.eye(2)
        A_n[f"z{n}", "l"] = -0.5
        # A_n.print(variables=variables)
        A_list.append(A_n)
    return A_0, A_list


def test_join_dicts():
    from poly_matrix import join_dicts

    a = {"x1": [1, 2, 3], "x2": [3]}
    b = {"x2": [1], "x3": [0]}
    output_test = {"x1": [1, 2, 3], "x2": [1, 3], "x3": [0]}
    output = sorted_dict(join_dicts(a, b))
    assert output == output_test

    a = {"x1": {"b": 3}, "x2": {"b": 4}}
    b = {"x2": {"c": 1}, "x3": {"b": 5}}
    output_test = {"x1": {"b": 3}, "x2": {"b": 4, "c": 1}, "x3": {"b": 5}}
    output = sorted_dict(join_dicts(a, b))
    assert output == join_dicts(a, b)


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


def test_blocks():
    """
    Reminder: Q is
    x1  1  2  3  4  7  0  0
        *  3  4  5  8  0  0
    x2  *  *  0  0  0  9  0
        *  *  *  0  0  10 0
    z1  *  *  *  *  3  0  0
    z2  *  *  *  *  *  2  0
     l  *  *  *  *  *  *  1
    """

    Q = get_Q()
    # get bigger blocks
    """
    X1Z1 = 1  2  7
           2  3  8
           7  8  3
    X2Z2 = 0  0  9
           0  0  10
           9  10 2
    """
    X1Z1, X2Z2 = Q.get_block_matrices(
        [
            (["x1", "z1"], ["x1", "z1"]),
            (["x2", "z2"], ["x2", "z2"]),
        ]
    )
    np.testing.assert_allclose(
        X1Z1, np.r_[np.c_[1, 2, 7], np.c_[2, 3, 8], np.c_[7, 8, 3]]
    )
    np.testing.assert_allclose(
        X2Z2, np.r_[np.c_[0, 0, 9], np.c_[0, 0, 10], np.c_[9, 10, 2]]
    )

    # get standard blocks
    X1, X2, Z1, Z2, L = Q.get_block_matrices(None)  # get all diagonal blocks
    np.testing.assert_allclose(X1, np.r_[np.c_[1, 2], np.c_[2, 3]])
    np.testing.assert_allclose(X2, np.zeros((2, 2)))
    np.testing.assert_allclose(Z1, np.array([[3]]))
    np.testing.assert_allclose(Z2, np.array([[2]]))
    np.testing.assert_allclose(L, np.array([[1]]))

    X1, X2, Z2 = Q.get_block_matrices(
        [("x1", "x1"), ("x2", "x2"), ("z2", "z2")]
    )  # get diagonal blocks x1, x2 and z2
    np.testing.assert_allclose(X1, np.r_[np.c_[1, 2], np.c_[2, 3]])
    np.testing.assert_allclose(X2, np.zeros((2, 2)))
    np.testing.assert_allclose(Z2, np.array([[2]]))

    X12, X2Z2, X1Z1 = Q.get_block_matrices([("x1", "x2"), ("x2", "z2"), ("x1", "z1")])
    np.testing.assert_allclose(X2Z2, np.c_[[9, 10]])
    np.testing.assert_allclose(X12, np.r_[np.c_[3, 4], np.c_[4, 5]])
    np.testing.assert_allclose(X1Z1, np.c_[[7, 8]])


def test_Q():
    Q = get_Q(test=True)
    print("Q", Q)


def test_Ai():
    A_0, A_list = get_Ai(test=True)
    print("A_0")
    print(A_0)
    print("A_i:")
    [print(A_i) for A_i in A_list]


def test_operations():

    # mat1 =
    # 1 0
    # 0 2
    mat1 = PolyMatrix()
    mat1[0, 0] = 1.0
    mat1[1, 1] = 2.0
    assert mat1.nnz == 2

    mat0 = PolyMatrix()
    mat0 += mat0 + mat1
    assert mat0.nnz == 2

    # mat2 =
    # 3 1
    # 1 0
    mat2 = PolyMatrix()
    mat2[0, 0] = 3.0
    mat2[0, 1] = 1.0
    assert mat2.nnz == 3

    mat3 = mat1 + mat2
    assert mat3[0, 0] == 4
    assert mat3[1, 1] == 2
    assert mat3[0, 1] == 1
    assert mat3[1, 0] == 1
    assert mat3.nnz == 4

    mat4 = mat1 - mat2
    assert mat4[0, 0] == -2
    assert mat4[1, 1] == 2
    assert mat4[0, 1] == -1
    assert mat4[1, 0] == -1
    assert mat4.nnz == 4

    mat5 = mat1 * 2
    assert mat5[0, 0] == 2
    assert mat5[1, 1] == 4
    assert mat5[0, 1] is None
    assert mat5[1, 0] is None
    assert mat5.nnz == 2

    mat6 = mat2 * 2
    assert mat6[0, 0] == 6
    assert mat6[1, 1] is None
    assert mat6[0, 1] == 2
    assert mat6[1, 0] == 2
    assert mat6.nnz == 3

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

    # test that variable_dict gets updated correctly
    A_0, A_list = get_Ai()
    Test = A_0 + A_list[0]
    variable_dict = Test.generate_variable_dict_i(["l", "z1", "x1"])
    assert variable_dict == {
        "l": {"index": 0, "size": 1},
        "z1": {"index": 1, "size": 1},
        "x1": {"index": 2, "size": 2},
    }

    variables = Q.variable_dict_i
    H_mat = Q.get_matrix(variables)
    H = Q.copy()
    np.testing.assert_equal(H.get_matrix(variables).toarray(), H_mat.toarray())

    H_mat += A_0.get_matrix(variables) * 2.0
    H += A_0 * 2.0
    np.testing.assert_equal(H.get_matrix(variables).toarray(), H_mat.toarray())

    for A_n in A_list:
        H_mat += A_n.get_matrix(variables)
        H += A_n
        np.testing.assert_equal(H.get_matrix(variables).toarray(), H_mat.toarray())


def test_get_matrix():
    """
    Build matrix:
    x1    x2    l
    1  2  3  4  5
    2  7  8  9  10
    3  8
    4  9
    5  10
    """
    mat = PolyMatrix()
    mat["x1", "x1"] = np.c_[[1, 2], [2, 7]]
    mat["x2", "x1"] = np.c_[[3, 4], [8, 9]]
    mat["l", "x1"] = np.c_[5, 10]
    assert mat.nnz == 16

    M_test = np.c_[
        [1, 2, 3, 4, 5],
        [2, 7, 8, 9, 10],
        [3, 8, 0, 0, 0],
        [4, 9, 0, 0, 0],
        [5, 10, 0, 0, 0],
    ]
    M = mat.get_matrix().toarray()
    np.testing.assert_equal(M_test, M)

    Mpart_test = np.c_[[3, 8], [4, 9]]
    Mpart = mat.get_matrix((["x1"], ["x2"])).toarray()
    np.testing.assert_equal(Mpart_test, Mpart)

    Mpart_poly = mat.get_matrix((["x1"], ["x2"]), output_type="poly")
    Mpart = Mpart_poly.get_matrix((["x1"], ["x2"])).toarray()
    np.testing.assert_equal(Mpart_test, Mpart)


def test_reorder():
    Q = get_Q()

    variables = list(Q.variable_dict_i.keys())

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


def test_scaling():
    d = 5
    filler = np.arange(d**2).reshape((d, d))
    filler += filler.T
    for N in np.logspace(1, 3, 3).astype(int):
        print(f"filling {N}...", end="")
        mat = PolyMatrix()

        for n in range(N):
            mat[f"x{n}", f"x{n}"] = filler
            mat[f"x{n}", f"z{n}"] = filler[:, 0]

        print("getting matrix...", end="")

        mat.get_matrix(verbose=True)
        print("...done")


def test_multiply():
    """
    Given the standard Q matrix:
        x1    x2    z1 z2 l
    x1  1  2  3  4  7  0  0
        *  3  4  5  8  0  0
    x2  *  *  0  0  0  9  0
        *  *  *  0  0  10 0
    z1  *  *  *  *  3  0  0
    z2  *  *  *  *  *  2  0
     l  *  *  *  *  *  *  1

    Perform the following multiplication:
    S = A B A.T
          z1  z2        z1 z2        x1    x2
    = x1  7   0    z1   3  0     z1  7  8  0  0
          8   0    z2   0  2     z2  0  0  9  10
      x2  0   9
          0   10
            x1              x2
    = x1  3*7*7   3*7*8   0      0
          3*7*8   3*8*8   0      0
      x2  0       0       2*9*9  2*9*10
          0       0       2*9*10 2*10*10
    """
    Q = get_Q()

    A = Q.get_matrix_poly((["x1", "x2"], ["z1", "z2"]))
    B = Q.get_matrix_poly(variables=["z1", "z2"])

    AT = A.transpose()
    S = A.multiply(B.multiply(AT))
    np.testing.assert_allclose(
        S["x1", "x1"], np.c_[np.r_[3 * 7 * 7, 3 * 7 * 8], np.r_[3 * 7 * 8, 3 * 8 * 8]]
    )
    np.testing.assert_allclose(
        S["x2", "x2"],
        np.c_[np.r_[2 * 9 * 9, 2 * 9 * 10], np.r_[2 * 9 * 10, 2 * 10 * 10]],
    )


if __name__ == "__main__":
    import sys

    test_multiply()
    test_get_matrix()

    test_join_dicts()
    test_blocks()
    test_scaling()
    test_addition()
    test_reorder()
    test_operations()
    test_Ai()
    test_Q()

    print("all tests passed")
