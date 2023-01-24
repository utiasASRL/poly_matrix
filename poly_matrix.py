from copy import deepcopy

import warnings

import numpy as np


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
            assert not key_j in self.adjacency_i[key_i]
            self.adjacency_i[key_i].append(key_j)
        else:
            self.adjacency_i[key_i] = [key_j]

            assert not key_i in self.matrix.keys()
            self.matrix[key_i] = {}

        if key_j in self.adjacency_j.keys():
            assert not key_i in self.adjacency_j[key_j]
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
        new_i = False
        new_j = False

        key_i, key_j = key_pair

        # make sure val is a float-ndarray
        if type(val) != np.ndarray:
            val = np.array(val, dtype=float)
            if val.ndim == 0:
                val = val.reshape((-1, 1))  # default to row vector
        else:
            if val.dtype != float:
                val = val.astype(float)

        if key_i not in self.variable_dict.keys():
            self.add_variable(key_i, val.shape[0])

        if key_j not in self.variable_dict.keys():
            self.add_variable(key_j, val.shape[1])

        # TODO(FD) figure out if there is a better way than
        # using new_i, new_j variables.
        if key_i not in self.adjacency_i.keys():
            new_i = True
        else:
            # make sure the dimensions of new block are consistent with
            # previously inserted blocks.
            assert val.shape[0] == self.variable_dict[key_i]["size"]

        if key_j not in self.adjacency_j.keys():
            new_j = True
        else:
            # make sure the dimensions of new block are consistent with
            # previously inserted blocks.
            assert val.shape[1] == self.variable_dict[key_j]["size"]

        # only add variables if either is new.
        if new_i or new_j:
            self.add_key_pair(key_i, key_j)

        if key_i == key_j:
            # main-diagonal blocks: make sure values are is symmetric
            np.testing.assert_equal(val, val.T)
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
            self.start = 0
            new_variable_dict = {}
            for v in variables:
                size = self.variable_dict[v]["size"]
                new_variable_dict[v] = {"size": size, "start": self.start}
                self.start += size
            self.variable_dict = new_variable_dict

        if type(variables) is dict:
            self.variable_dict = variables

    def copy(self):
        return deepcopy(self)

    def print(self, variables=None):
        print(self.__repr__(variables=variables))

    def get_matrix(self, variables=None):
        """Return a sparse matrix in COO-format.

        :param variables: list of variable order to use (defaults to order in self.variable_dict)

        """
        if variables:
            assert len(variables) == len(
                self.variable_dict
            ), "inconsistent number of variables. Use reorder first"
            variable_dict = {v: self.variable_dict[v] for v in variables}
        else:
            variable_dict = self.variable_dict

        import scipy.sparse as sp

        i_list = []
        j_list = []
        data_list = []
        for i, (key_i, dict_i) in enumerate(variable_dict.items()):
            for j, (key_j, dict_j) in enumerate(variable_dict.items()):
                if j < i:
                    continue

                try:
                    self.matrix[key_i][key_j]
                except:
                    continue

                ii, jj = np.meshgrid(range(dict_i["size"]), range(dict_j["size"]))
                i_list += (ii.flatten() + dict_i["start"]).tolist()
                j_list += (jj.flatten() + dict_j["start"]).tolist()
                data_list += self.matrix[key_i][key_j].flatten().tolist()

                if j > i:  # off-diagonal blocks, repeat symmetrically
                    j_list += (ii.flatten() + dict_i["start"]).tolist()
                    i_list += (jj.flatten() + dict_j["start"]).tolist()
                    data_list += self.matrix[key_i][key_j].flatten().tolist()
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

    def __repr__(self, variables=None):
        """Called by the print() function"""
        import pandas

        if not variables:
            variables = self.variable_dict.keys()

        df = pandas.DataFrame(columns=variables, index=variables)
        df.update(self.matrix)
        return df.to_string()

    def __add__(self, other, inplace=False):
        if inplace:
            res = self
        else:
            res = self.copy()

        if type(other) == PolyMatrix:
            for key_i in other.adjacency_i.keys():
                for key_j in other.adjacency_i[key_i]:
                    if other[key_i, key_j] is not None:
                        if res[key_i, key_j] is None:

                            # important: add new elements to adjacency array
                            res.add_key_pair(key_i, key_j)

                            # need deepcopy because other may be array
                            res.matrix[key_i][key_j] = deepcopy(other[key_i, key_j])
                        else:
                            res.matrix[key_i][key_j] += deepcopy(other[key_i, key_j])
        else:
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
