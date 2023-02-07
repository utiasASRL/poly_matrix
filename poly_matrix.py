from copy import deepcopy

import warnings

import numpy as np
import scipy.sparse as sp


def join_dicts(a, b):
    """Join two dicts of lists or two dicts of dicts.

    Examples
    a: {"x1":[1, 2, 3], "x2":[3]},
    b: {"x2": [1], "x3":[0]}
    returns: {"x1":[1, 2, 3], "x2":[3, 1], "x3": [0]}

    a: {"x1":{"b": 3}, "x2":{"b": 4}},
    b: {"x2":{"c": 1}, "x3":{"b": 5}}
    returns: {"x1":{"b": 3}, "x2":{"b": 4, "c": 1}, "x3": {"b": 5}}
    """
    unique_keys = set(list(a.keys()) + list(b.keys()))
    dict_out = {}
    for key in unique_keys:
        if (type(a.get(key, None)) == list) or (type(b.get(key, None)) == list):
            dict_out[key] = list(set(a.get(key, [])).union(set(b.get(key, []))))
        elif (type(a.get(key, None)) == dict) or (type(b.get(key, None)) == dict):
            dict_out[key] = a.get(key, {})
            dict_out[key].update(b.get(key, {}))
    return dict_out


def get_shape(variable_dict_i, variable_dict_j):
    last_elemi = next(reversed(variable_dict_i.values()))
    last_elemj = next(reversed(variable_dict_j.values()))
    return (
        last_elemi["index"] + last_elemi["size"],
        last_elemj["index"] + last_elemi["size"],
    )


class PolyMatrix(object):
    def __init__(self):
        self.matrix = {}

        self.last_var_index = 0
        self.nnz = 0

        # dictionary of form {variable-key: {size: variable-size, index: variable-start-index}}
        # TODO(FD) consider replacing with NamedTuple
        self.variable_dict = {}

        # TODO(FD) technically, adjacency_i has redundant information, since
        # self.matrix.keys() could be used. Consider removing it (for now,
        # it is kept for analogy with adjacency_j).
        # adjacency_j allows for fast starting of the adjacency variables of j.
        self.adjacency_i = {}
        self.adjacency_j = {}

    def __getitem__(self, key):
        key_i, key_j = key
        try:
            return self.matrix[key_i][key_j]
        except:
            None

    def size(self):
        return self.last_var_index

    def add_variable(self, key, size):
        self.variable_dict[key] = {"size": size, "index": self.last_var_index}
        self.last_var_index += size

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
            # print(f"Warning: converting {key_pair}'s value to column vector.")
            val = val.reshape((-1, 1))  # default to column vector

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
        self.add_key_pair(key_i, key_j)

        if key_i == key_j:
            # main-diagonal blocks: make sure values are is symmetric
            np.testing.assert_allclose(val, val.T, rtol=1e-10)

            self.matrix[key_i][key_j] = deepcopy(val)
            self.nnz += val.size
        elif symmetric:
            # fill symmetrically (but set symmetric to False to not end in infinite loop)
            self.matrix[key_i][key_j] = deepcopy(val)
            self.nnz += val.size

            self.__setitem__([key_j, key_i], val.T, symmetric=False)
        else:
            self.matrix[key_i][key_j] = deepcopy(val)
            self.nnz += val.size

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

    def get_shape(self):
        return self.get_size(), self.get_size()

    def generate_variable_dict(self, variables=None):
        """Regenerate last_var_index using new ordering."""
        if variables is None:
            variables = list(self.matrix.keys())

        last_var_index = 0
        variable_dict = {}
        for key in variables:
            size = self.variable_dict[key]["size"]
            variable_dict[key] = {
                "index": last_var_index,
                "size": size,
            }
            last_var_index += size
        return variable_dict

    def get_variables(self, key=None):
        """Return variable names.
        :param key: Which names to extract, either of type
            - None: all names, as ordered in self.variable_dict
            - str: return all variable names indexing with this string
        """
        if key is None:
            return list(self.variable_dict.keys())
        else:
            return sorted([v for v in self.variable_dict.keys() if v.startswith(key)])

    def get_nnz(self, variable_dict_i=None, variable_dict_j=None):
        """Get number of non-zero entries in sumatrix chosen by variable_dict_i, variable_dict_j."""
        if variable_dict_i is None:
            variable_dict_i = self.variable_dict
        if variable_dict_j is None:
            variable_dict_j = self.variable_dict

        # this is much faster than below
        nnz = 0
        for key_i in set(variable_dict_i.keys()).intersection(self.matrix.keys()):
            for key_j in set(variable_dict_j.keys()).intersection(self.matrix[key_i]):
                nnz += self.matrix[key_i][key_j].size
        return nnz
        # for key_i, dict_i in variable_dict_i.items():
        #    if key_i not in self.matrix.keys():
        #        continue
        #    for key_j, dict_j in variable_dict_j.items():
        #        if key_j not in self.matrix[key_i].keys():
        #            continue
        #        nnz += self.matrix[key_i][key_j].size
        # return nnz

    def get_matrix(self, variables=None, sparsity_type="coo", verbose=False):
        if sparsity_type == "none":
            return self.get_matrix_dense(variables=variables, verbose=verbose)
        else:
            return self.get_matrix_sparse(
                variables=variables, sparsity_type=sparsity_type, verbose=verbose
            )

    def get_matrix_dense(self, variables, verbose=False):
        """Return a small submatrix in dense format

        :param variables: same as in self.get_matrix_sparse, but None is not allowed
        """
        assert variables is not None
        if type(variables) == "list":
            variable_dict_i = self.generate_variable_dict(variables)
            variable_dict_j = variable_dict_i
        elif type(variables) == tuple:
            variable_dict_i = self.generate_variable_dict(variables[0])
            variable_dict_j = self.generate_variable_dict(variables[1])

        shape = get_shape(variable_dict_i, variable_dict_j)
        matrix = np.zeros(shape)

        for key_i in set(variable_dict_i.keys()).intersection(self.matrix.keys()):
            for key_j in set(variable_dict_j.keys()).intersection(self.matrix[key_i]):
                dict_j = variable_dict_j[key_j]
                dict_i = variable_dict_i[key_i]

                shape = (dict_i["size"], dict_j["size"])

                # We are not sure if values are stored in [i, j] or [j, i],
                # so we check, and take transpose if necessary.

                if key_j in self.matrix.get(key_i, {}).keys():
                    values = self.matrix[key_i][key_j]
                elif key_i in self.matrix.get(key_j, {}).keys():
                    values = self.matrix[key_j][key_i].T
                else:
                    values = np.zeros(shape)

                matrix[
                    dict_i["index"] : dict_i["index"] + dict_i["size"],
                    dict_j["index"] : dict_j["index"] + dict_j["size"],
                ] = values
        return matrix

    def get_matrix_sparse(self, variables=None, sparsity_type="coo", verbose=False):
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
                variable_dict_i = self.generate_variable_dict(variables[0])
                variable_dict_j = self.generate_variable_dict(variables[1])
        else:
            variable_dict_i = self.variable_dict
            variable_dict_j = variable_dict_i

        import time

        t1 = time.time()
        nnz = self.get_nnz(variable_dict_i, variable_dict_j)
        if verbose:
            print(f"Finding nonzero elements took {time.time() - t1:.2}s.")

        t1 = time.time()
        i_list = np.empty(nnz, dtype=int)
        j_list = np.empty(nnz, dtype=int)
        data_list = np.empty(nnz, dtype=float)
        index = 0

        # no difference in speed between below and current implementation.
        # i_list = []
        # j_list = []
        # data_list = []
        # for key_i, dict_i in variable_dict_i.items():
        #    if not key_i in self.matrix.keys():
        #        continue
        #    for key_j, dict_j in variable_dict_j.items():
        #        if not key_j in self.matrix[key_i].keys():
        #            continue

        t1 = time.time()
        for key_i in set(variable_dict_i.keys()).intersection(self.matrix.keys()):
            for key_j in set(variable_dict_j.keys()).intersection(self.matrix[key_i]):
                dict_j = variable_dict_j[key_j]
                dict_i = variable_dict_i[key_i]

                # We are not sure if values are stored in [i, j] or [j, i],
                # so we check, and take transpose if necessary.
                if key_j in self.matrix.get(key_i, {}).keys():
                    values = self.matrix[key_i][key_j]
                elif key_i in self.matrix.get(key_j, {}).keys():
                    values = self.matrix[key_j][key_i].T
                else:
                    continue

                jj, ii = np.meshgrid(range(dict_j["size"]), range(dict_i["size"]))
                i_list[index : index + ii.size] = ii.flatten() + dict_i["index"]
                j_list[index : index + ii.size] = jj.flatten() + dict_j["index"]
                data_list[index : index + ii.size] = values.flatten()
                index += ii.size

                # i_list += (ii.flatten() + dict_i["index"]).tolist()
                # j_list += (jj.flatten() + dict_j["index"]).tolist()
                # data_list += values.flatten().tolist()

        if verbose:
            print(f"Filling took {time.time() - t1:.2}s.")

        assert index == nnz
        last_elemi = next(reversed(variable_dict_i.values()))
        size_i = last_elemi["index"] + last_elemi["size"]
        last_elemj = next(reversed(variable_dict_j.values()))
        size_j = last_elemj["index"] + last_elemj["size"]
        shape = (size_i, size_j)

        t1 = time.time()
        if sparsity_type == "coo":
            mat = sp.coo_matrix((data_list, (i_list, j_list)), shape=shape)
        elif sparsity_type == "csr":
            mat = sp.csr_matrix((data_list, (i_list, j_list)), shape=shape)
        elif sparsity_type == "csc":
            mat = sp.csc_matrix((data_list, (i_list, j_list)), shape=shape)
        else:
            raise ValueError(f"Unknown matrix type {sparsity_type}")

        if verbose:
            print(f"Filling took {time.time() - t1:.2}s.")

        return mat

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

    def get_block_matrices(self, key_list=None):
        if key_list is None:
            key_list = list(zip(self.variable_dict.keys(), self.variable_dict.keys()))

        # example 1: key_list = [("x1", "x2"), ("x1", "x3"),...]
        # example 2: key_list = [(["x1", "z1"], ["x2", "z2"]), (["x1", "z1"], ["x3", "z3"])...]
        blocks = []
        for key_pair in key_list:
            if (type(key_pair) is tuple) and (type(key_pair[0]) is list):
                key_i, key_j = key_pair
                assert type(key_j) is list, "Mixed types not allowed"
                # example 2: append submatrices
                blocks.append(
                    self.get_matrix(variables=(key_i, key_j), sparsity_type="none")
                )
            else:
                if type(key_pair) is tuple:
                    # example 1: append H["x1", "x2"], H["x1", "x3"], etc.
                    key_i, key_j = key_pair
                else:
                    key_i = key_j = key_pair

                try:
                    blocks.append(self.matrix[key_i][key_j])
                except:
                    i_size = self.variable_dict[key_i]["size"]
                    j_size = self.variable_dict[key_j]["size"]
                    blocks.append(np.zeros((i_size, j_size)))
        return blocks

    def __repr__(self, variables=None, binary=False):
        """Called by the print() function"""
        output = f"Sparse polymatrix of shape {self.size(), self.size()}\n"
        output += f"Number of nnz: {self.get_nnz()}\n\n"

        if self.size() > 100:
            return output

        import pandas

        if not variables:
            variables = self.variable_dict.keys()

        df = pandas.DataFrame(columns=variables, index=variables)
        df.update(self.matrix)
        if (len(df) > 10) or binary:
            df = df.notnull().astype("int")
        else:
            df.fillna(0, inplace=True)
        return output + df.to_string()

    def __add__(self, other, inplace=False):
        if inplace:
            res = self
        else:
            res = deepcopy(self)

        if type(other) == PolyMatrix:
            # add two different polymatrices
            res.adjacency_i = join_dicts(other.adjacency_i, res.adjacency_i)
            res.adjacency_j = join_dicts(other.adjacency_j, res.adjacency_j)
            res.variable_dict = join_dicts(other.variable_dict, res.variable_dict)
            for key_i in res.adjacency_i.keys():
                for key_j in res.adjacency_i[key_i]:

                    other_nnz = (key_i in other.matrix.keys()) and (
                        key_j in other.matrix[key_i].keys()
                    )
                    res_nnz = (key_i in res.matrix.keys()) and (
                        key_j in res.matrix[key_i].keys()
                    )
                    assert (
                        other_nnz or res_nnz
                    )  # either has to be true, or this pair should not be in the adjacency list.

                    if res_nnz and not other_nnz:  # add nothing to nonzero
                        continue
                    elif res_nnz and other_nnz:  # add nonzero to nonzero
                        new_mat = res.matrix[key_i][key_j] + other.matrix[key_i][key_j]
                        res.matrix[key_i][key_j] = deepcopy(new_mat)
                    elif (not res_nnz) and other_nnz:  # add nonzero to zero
                        new_mat = other.matrix[key_i][key_j]
                        if key_i in res.matrix.keys():
                            res.matrix[key_i][key_j] = deepcopy(new_mat)
                        else:
                            res.matrix[key_i] = {key_j: deepcopy(new_mat)}
                        res.nnz += new_mat.size
        else:
            # simply add constant to all non-zero elements
            for key_i in res.adjacency_i.keys():
                for key_j in res.adjacency_i[key_i]:
                    if other[key_i, key_j] is not None:
                        res.matrix[key_i][key_j] += other

        # regenerate variable dict to fix order.
        res.variable_dict = res.generate_variable_dict()
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
