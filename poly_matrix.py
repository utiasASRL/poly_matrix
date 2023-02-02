from copy import deepcopy

import warnings

import numpy as np


def join_dicts(a, b):
    """Join two dicts of lists.

    Example
    a: {"x1":[1, 2, 3], "x2":[3]},
    b: {"x2": [1], "x3":[0]}
    returns: {"x1":[1, 2, 3], "x2":[3, 1], "x3": [0]}
    """
    unique_keys = set(list(a.keys()) + list(b.keys()))
    dict = {}
    for key in unique_keys:
        dict[key] = list(set(a.get(key, [])).union(set(b.get(key, []))))
    return dict


class PolyMatrix(object):
    def __init__(self):
        self.matrix = {}

        self.start = 0

        # dictionary of form {variable-key: {size: variable-size, start: variable-start-index}}
        # TODO(FD) consider replacing with NamedTuple
        self.variable_dict = {}

        # TODO(FD) technically, adjacency_i has redundant information, since
        # self.matrix.keys() could be used. Consider removing it (for now,
        # it is kept for analogy with adjacency_j).
        # adjacency_j allows for fast indexing of the adjacency variables of j.
        self.adjacency_i = {}
        self.adjacency_j = {}

    def __getitem__(self, key):
        key_i, key_j = key
        try:
            return self.matrix[key_i][key_j]
        except:
            None

    def add_variable(self, key, size):
        self.variable_dict[key] = {"size": size, "start": self.start}
        self.start += size

    def add_key_pair(self, key_i, key_j):
        if key_i in self.adjacency_i.keys():
            if not (key_j in self.adjacency_i[key_i]):
                self.adjacency_i[key_i].append(key_j)
        else:
            self.adjacency_i[key_i] = [key_j]

            assert not key_i in self.matrix.keys()
            self.matrix[key_i] = {}

        if key_j in self.adjacency_j.keys():
            if not (key_i in self.adjacency_j[key_j]):
                self.adjacency_j[key_j].append(key_i)
        else:
            self.adjacency_j[key_j] = [key_i]

    def __setitem__(self, key_pair, val, symmetric=True):
        """
        :param key_pair: pair of variable names, e.g. ('row1', 'col1'), whose block will be populated
        :param val: values at that block
        :param symmetric: fill the matrix symmetrically, defaults to True.
        """
        # flags to indicate if this i or j index already exists in the matrix,
        # if yes dimensions are checked.
        key_i, key_j = key_pair

        # make sure val is a float-ndarray
        if type(val) != np.ndarray:
            val = np.array(val, dtype=float)
        else:
            if val.dtype != float:
                val = val.astype(float)

        if val.ndim < 2:
            val = val.reshape((-1, 1))  # default to row vector

        if key_i not in self.variable_dict.keys():
            self.add_variable(key_i, val.shape[0])

        if key_j not in self.variable_dict.keys():
            self.add_variable(key_j, val.shape[1])

        # make sure the dimensions of new block are consistent with
        # previously inserted blocks.
        if key_i in self.adjacency_i.keys():
            assert val.shape[0] == self.variable_dict[key_i]["size"]
        if key_j in self.adjacency_j.keys():
            assert val.shape[1] == self.variable_dict[key_j]["size"]

        # only add variables if either is new.
        self.add_key_pair(key_i, key_j)

        if key_i == key_j:
            # main-diagonal blocks: make sure values are is symmetric
            np.testing.assert_almost_equal(val, val.T)
            self.matrix[key_i][key_j] = deepcopy(val)
        elif symmetric:
            # fill symmetrically (but set symmetric to False to not end in infinite loop)
            self.matrix[key_i][key_j] = deepcopy(val)
            self.__setitem__([key_j, key_i], val.T, symmetric=False)
        else:
            self.matrix[key_i][key_j] = deepcopy(val)

    def reorder(self, variables=None):
        """Reinitiate variable dictionary, making sure all sizes are consistent"""
        if type(variables) is list:
            assert len(variables) == len(self.variable_dict)
            self.variable_dict = self.generate_variable_dict(variables)
        elif type(variables) is dict:
            self.variable_dict = variables

    def copy(self):
        return deepcopy(self)

    def print(self, variables=None, binary=False):
        print(self.__repr__(variables=variables, binary=binary))

    def generate_variable_dict(self, variables):
        """Regenerate start indices using new ordering."""
        start = 0
        variable_dict = {}
        for key in variables:
            size = self.variable_dict[key]["size"]
            variable_dict[key] = {
                "start": start,
                "size": size,
            }
            start += size
        return variable_dict

    def get_variables(self):
        return list(self.variable_dict.keys())

    def get_matrix(self, variables=None):
        """Return a sparse matrix in COO-format.

        :param variables: Can be any of the following:
            - list of variables to use, returns square matrix
            - tuple of (variables_i, variables_j), where both are lists. Returns any-size matrix
            - None: use self.variable_dict instead

        """
        if variables:
            if type(variables) == list:
                variable_dict_i = self.generate_variable_dict(variables)
                variable_dict_j = variable_dict_i
            elif type(variables) == tuple:
                # TODO(FD) continue here: need to change start according to sizes!
                variable_dict_i = self.generate_variable_dict(variables[0])
                variable_dict_j = self.generate_variable_dict(variables[1])
        else:
            variable_dict_i = self.variable_dict
            variable_dict_j = variable_dict_i

        import scipy.sparse as sp

        i_list = []
        j_list = []
        data_list = []
        for (key_i, dict_i) in variable_dict_i.items():
            for (key_j, dict_j) in variable_dict_j.items():

                # We are not sure if values are stored in [i, j] or [j, i],
                # so we check, and take transpose if necessary.
                if key_j in self.matrix.get(key_i, {}).keys():
                    values = self.matrix[key_i][key_j]
                elif key_i in self.matrix.get(key_j, {}).keys():
                    values = self.matrix[key_j][key_i].T
                else:
                    continue

                jj, ii = np.meshgrid(range(dict_j["size"]), range(dict_i["size"]))
                i_list += (ii.flatten() + dict_i["start"]).tolist()
                j_list += (jj.flatten() + dict_j["start"]).tolist()
                data_list += values.flatten().tolist()
        size = max(max(i_list) + 1, max(j_list) + 1)
        return sp.coo_matrix((data_list, (i_list, j_list)), shape=(size, size))

    def get_vector(self, variables=None, **kwargs):
        if variables:
            variable_dict = {v: self.variable_dict[v] for v in variables}
        else:
            variable_dict = self.variable_dict

        vector = []
        for var in variable_dict.keys():
            if not var in kwargs.keys():
                warnings.warn(f"{var} not in {variable_dict.keys()}")
            val = kwargs[var]
            vector += np.array([val]).flatten().tolist()
        return np.array(vector).astype(float)

    def get_vector_from_theta(self, theta):
        """Get vector using as input argument theta = [theta1; theta2; ...] = [x1, v1; x2, v2; ...]

        Assumes variable names are 'xi' for thetai and 'zi' for ||xi||^2, and 'l' for 1.
        """
        vector_dict = {}
        k = theta.shape[1]
        if k > 3:
            d = k // 2
        else:
            d = k
        assert d in [2, 3]

        for i, theta_i in enumerate(theta):
            assert len(theta_i) == k
            vector_dict[f"x{i}"] = theta_i
            vector_dict[f"z{i}"] = np.linalg.norm(theta_i[:d]) ** 2
        vector_dict["l"] = 1.0
        return self.get_vector(**vector_dict)

    def get_block_matrices(self, list_of_key_lists):
        output = []
        for key_list in list_of_key_lists:
            blocks = []
            for key_i, key_j in key_list:
                blocks.append(self.matrix[key_i][key_j])
            output.append(blocks)
        return output

    def __repr__(self, variables=None, binary=False):
        """Called by the print() function"""
        import pandas

        if not variables:
            variables = self.variable_dict.keys()

        df = pandas.DataFrame(columns=variables, index=variables)
        df.update(self.matrix)
        if (len(df) > 10) or binary:
            df = df.notnull().astype("int")
        else:
            df.fillna(0, inplace=True)
        return df.to_string()

    def __add__(self, other, inplace=False):
        if inplace:
            res = self
        else:
            res = self.copy()

        if type(other) == PolyMatrix:
            # add two different polymatrices
            res.adjacency_i = join_dicts(other.adjacency_i, res.adjacency_i)
            res.adjacency_j = join_dicts(other.adjacency_j, res.adjacency_j)
            for key_i in res.adjacency_i.keys():
                for key_j in res.adjacency_i[key_i]:

                    if key_i in res.matrix:
                        res.matrix[key_i][key_j] = deepcopy(
                            other.matrix.get(key_i, {}).get(key_j, 0)
                        ) + deepcopy(res.matrix.get(key_i, {}).get(key_j, 0))
                    else:
                        res.matrix[key_i] = {
                            key_j: deepcopy(other.matrix.get(key_i, {}).get(key_j, 0))
                            + deepcopy(res.matrix.get(key_i, {}).get(key_j, 0))
                        }
        else:
            # simply add constant to all non-zero elements
            for key_i in res.adjacency_i.keys():
                for key_j in res.adjacency_i[key_i]:
                    if other[key_i, key_j] is not None:
                        res.matrix[key_i][key_j] += other
        return res

    def __sub__(self, other):
        return self + (other * (-1))

    def __rmul__(self, scalar, inplace=False):
        """Overload a * M"""
        return self.__mul__(scalar, inplace)

    def __mul__(self, scalar, inplace=False):
        """Overload M * a"""
        assert np.ndim(scalar) == 0, "Multiplication with non-scalar not supported"
        if inplace:
            res = self
        else:
            res = self.copy()
        for key_i in res.adjacency_i.keys():
            for key_j in res.adjacency_i[key_i]:
                res.matrix[key_i][key_j] *= scalar
        return res

    def __iadd__(self, other):
        """Overload the += operation"""
        return self.__add__(other, inplace=True)

    def __imul__(self, other):
        """Overload the *= operation"""
        return self.__mul__(other, inplace=True)


if __name__ == "__main__":
    mat1 = PolyMatrix()
    mat1[0, 0] = 1.0
    mat1[1, 1] = 2.0

    mat2 = PolyMatrix()
    mat2[0, 0] = 3.0

    print("mat1")
    mat1.print([0, 1])
    print("mat2")
    mat2.print([0, 1])

    print(mat1 + mat2)
    print(mat1 - mat2)
    print(mat1 * 2)
    print(2 * mat2)
