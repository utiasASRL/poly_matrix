from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.linalg import issymmetric


def sorted_dict(dict_):
    return dict(sorted(dict_.items(), key=lambda val: val[0]))


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
    a = deepcopy(a)
    b = deepcopy(b)

    all_keys = list(a.keys()) + list(b.keys())
    dict_out = dict(zip(all_keys, [None] * len(all_keys)))
    for key in dict_out.keys():
        if (type(a.get(key, None)) == list) or (type(b.get(key, None)) == list):
            dict_out[key] = list(set(a.get(key, [])).union(set(b.get(key, []))))
        elif (type(a.get(key, None)) == dict) or (type(b.get(key, None)) == dict):
            dict_out[key] = a.get(key, {})
            dict_out[key].update(b.get(key, {}))
        else:
            if (key in a) and (key in b):
                assert a[key] == b[key]
            dict_out[key] = a.get(key, None) if a.get(key, None) else b.get(key, None)
    return dict_out


def generate_indices(variable_dict):
    indices = {}
    i = 0
    for key, size in variable_dict.items():
        indices[key] = i
        i += size
    return indices


def get_size(variable_dict):
    return np.sum([val for val in variable_dict.values()])


def get_shape(variable_dict_i, variable_dict_j):
    i_size = get_size(variable_dict_i)
    j_size = get_size(variable_dict_j)
    return i_size, j_size


class PolyMatrix(object):
    SPARSE_OUTPUT_TYPES = ["coo", "csr", "csc"]
    MATRIX_OUTPUT_TYPES = SPARSE_OUTPUT_TYPES + ["poly", "dense"]

    def __init__(self, symmetric=True):
        self.matrix = {}

        self.last_var_i_index = 0
        self.last_var_j_index = 0
        self.nnz = 0

        self.symmetric = symmetric

        # dictionary of form {variable-key : variable-size}
        # TODO(FD) consider replacing with NamedTuple
        self.variable_dict_i = {}
        self.variable_dict_j = {}

        # TODO(FD) technically, adjacency_i has redundant information, since
        # self.matrix.keys() could be used. Consider removing it (for now,
        # it is kept for analogy with adjacency_j).
        # adjacency_j allows for fast starting of the adjacency variables of j.
        self.adjacency_i = {}
        self.adjacency_j = {}

        self.shape = (0, 0)

    def __getitem__(self, key):
        key_i, key_j = key
        try:
            return self.matrix[key_i][key_j]
        except:
            # TODO(FD) below is probably not the best solution, but it works
            try:
                size = (self.variable_dict_i[key_i], self.variable_dict_j[key_j])
                if size == (1, 1):
                    return 0
                else:
                    return np.zeros(size)
            except KeyError:
                return 0

    def get_size(self):
        return max(self.last_var_i_index, self.last_var_j_index)

    def add_variable_i(self, key, size):
        self.variable_dict_i[key] = size
        self.last_var_i_index += size

    def add_variable_j(self, key, size):
        self.variable_dict_j[key] = size
        self.last_var_j_index += size

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

    def __setitem__(self, key_pair, val, symmetric=None):
        """
        :param key_pair: pair of variable names, e.g. ('row1', 'col1'), whose block will be populated
        :param val: values at that block
        :param symmetric: fill the matrix symmetrically, defaults to True.
        """
        if symmetric is None:
            symmetric = self.symmetric

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

        if key_i not in self.variable_dict_i:
            self.add_variable_i(key_i, val.shape[0])

        if key_j not in self.variable_dict_j:
            self.add_variable_j(key_j, val.shape[1])

        # make sure the dimensions of new block are consistent with
        # previously inserted blocks.
        if key_i in self.adjacency_i:
            assert val.shape[0] == self.variable_dict_i[key_i]
        if key_j in self.adjacency_j:
            assert val.shape[1] == self.variable_dict_j[key_j]
        self.add_key_pair(key_i, key_j)

        if key_i == key_j:
            # main-diagonal blocks: make sure values are symmetric
            if not issymmetric(val, rtol=1e-10):
                raise ValueError(
                    f"Input Matrix for keys: ({key_i},{key_j}) is not symmetric"
                )

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

        # needs this needs to be updated
        self.shape = None

    def reorder(self, variables=None):
        """Reinitiate variable dictionary, making sure all sizes are consistent"""
        if type(variables) is list:
            assert len(variables) == len(self.variable_dict_i)
            self.variable_dict_i = self.generate_variable_dict_i(variables)
            self.variable_dict_j = self.generate_variable_dict_j(variables)
        elif type(variables) is dict:
            self.variable_dict_i = variables
            self.variable_dict_j = variables

    def copy(self):
        return deepcopy(self)

    def print(self, variables=None, binary=False):
        print(self.__repr__(variables=variables, binary=binary))

    def get_shape(self):
        if self.shape is None:
            self.shape = get_shape(self.variable_dict_i, self.variable_dict_j)
        return self.shape

    def generate_variable_dict_i(self, variables=None):
        """Regenerate last_var_index using new ordering."""
        if variables is None:
            variables = list(self.adjacency_i.keys())
        return self._generate_variable_dict(variables, self.variable_dict_i)

    def generate_variable_dict_j(self, variables=None):
        """Regenerate last_var_index using new ordering."""
        if variables is None:
            variables = list(self.adjacency_j.keys())
        return self._generate_variable_dict(variables, self.variable_dict_j)

    def _generate_variable_dict(self, variables, variable_dict):
        if type(variables) == dict:
            return {key: variable_dict.get(key, variables[key]) for key in variables}
        else:
            return {key: variable_dict[key] for key in variables}

    def get_variables(self, key=None):
        """Return variable names starting with key.
        :param key: Which names to extract, either of type
            - None: all names, as ordered in self.variable_dict
            - str: return all variable names indexing with this string
        """
        all_keys = list(
            set(self.variable_dict_i.keys()).union(self.variable_dict_j.keys())
        )
        if key is None:
            return all_keys
        else:
            dict_ = {int(v.split(key)[1]): v for v in all_keys if v.startswith(key)}
            return list(sorted_dict(dict_).values())

    # TODO(FD): there is probably a much cleaner way of doing this -- basically,
    # keeping track of N somewhere else. For now, this is a quick hack.
    def get_max_index(self):
        max_ = 0
        for key in self.variable_dict.keys():
            if type(key) == "str":
                # works for keys of type 'x10'
                max_ = max(max_, int(key[1:])[0])
            else:
                max_ = max(max_, key)
        return max_

    def get_nnz(self, variable_dict_i=None, variable_dict_j=None):
        """Get number of non-zero entries in sumatrix chosen by variable_dict_i, variable_dict_j."""
        if variable_dict_i is None:
            variable_dict_i = self.variable_dict_i
        if variable_dict_j is None:
            variable_dict_j = self.variable_dict_j

        # this is much faster than below
        nnz = 0
        for key_i in set(variable_dict_i.keys()).intersection(self.matrix.keys()):
            for key_j in set(variable_dict_j.keys()).intersection(self.matrix[key_i]):
                nnz += variable_dict_i[key_i] * variable_dict_j[key_j]
        return nnz
        # for key_i, dict_i in variable_dict_i.items():
        #    if key_i not in self.matrix.keys():
        #        continue
        #    for key_j, dict_j in variable_dict_j.items():
        #        if key_j not in self.matrix[key_i].keys():
        #            continue
        #        nnz += self.matrix[key_i][key_j].size
        # return nnz

    def get_matrix(self, variables=None, output_type="csc", verbose=False):
        """Get the submatrix defined by variables.

        :param variables: Can be any of the following:
            - dict or list of variables to use, returns square matrix. If list is given, all keys must actually be
              in the matrix, otherwise the size of the matrix can't be determined.
            - tuple of (variables_i, variables_j), where both are lists. Returns any-size matrix
            - None: use self.variable_dict instead
        """
        assert output_type in self.MATRIX_OUTPUT_TYPES

        if output_type == "dense":
            return self.get_matrix_dense(variables=variables, verbose=verbose)
        elif output_type == "poly":
            return self.get_matrix_poly(variables=variables, verbose=verbose)
        elif output_type in self.SPARSE_OUTPUT_TYPES:
            return self.get_matrix_sparse(
                variables=variables, output_type=output_type, verbose=verbose
            )
        else:
            raise ValueError(output_type)

    def toarray(self, variables=None):
        return self.get_matrix_sparse(variables=variables).toarray()

    def get_matrix_poly(self, variables, verbose=False):
        """Return a PolyMatrix submatrix.

        :param variables: same as in self.get_matrix, but None is not allowed
        """
        assert variables is not None
        if type(variables) in [list, set]:
            variable_dict_i = self.generate_variable_dict_i(variables)
            variable_dict_j = self.generate_variable_dict_j(variables)
            symmetric = True
        elif type(variables) is tuple:
            variable_dict_i = self.generate_variable_dict_i(variables[0])
            variable_dict_j = self.generate_variable_dict_j(variables[1])
            symmetric = False

        out_matrix = PolyMatrix(symmetric=symmetric)
        for key_i in set(variable_dict_i.keys()).intersection(self.matrix.keys()):
            for key_j in set(variable_dict_j.keys()).intersection(self.matrix[key_i]):
                # We are not sure if values are stored in [i, j] or [j, i],
                # so we check, and take transpose if necessary.
                if key_j in self.matrix.get(key_i, {}).keys():
                    values = self.matrix[key_i][key_j]
                elif key_i in self.matrix.get(key_j, {}).keys():
                    values = self.matrix[key_j][key_i].T
                else:
                    continue
                    # values = np.zeros(shape)

                out_matrix[key_i, key_j] = values
                continue

                # TODO(FD): below is fairly standard and should be put in a function.
                if key_i in out_matrix.matrix.keys():
                    out_matrix.matrix[key_i][key_j] = values
                else:
                    out_matrix.matrix[key_i] = {key_j: values}
                    out_matrix.adjacency_i[key_i] = [key_j]
                    if key_i not in out_matrix.adjacency_j.get(key_j, []):
                        out_matrix.adjacency_j[key_j] = [key_i]
                    else:
                        out_matrix.adjacency_j[key_j].append(key_i)

        out_matrix.variable_dict_i = variable_dict_i
        out_matrix.variable_dict_j = variable_dict_j
        return out_matrix

    def get_matrix_dense(self, variables, verbose=False):
        """Return a small submatrix in dense format

        :param variables: same as in self.get_matrix, but None is not allowed
        """
        assert variables is not None
        if type(variables) == list:
            variable_dict_i = self.generate_variable_dict_i(variables)
            variable_dict_j = self.generate_variable_dict_j(variables)
        elif type(variables) == tuple:
            variable_dict_i = self.generate_variable_dict_i(variables[0])
            variable_dict_j = self.generate_variable_dict_j(variables[1])

        shape = get_shape(variable_dict_i, variable_dict_j)
        matrix = np.zeros(shape)

        index_i = 0
        for key_i, size_i in variable_dict_i.items():
            index_j = 0
            for key_j, size_j in variable_dict_j.items():
                shape = (size_i, size_j)

                # We are not sure if values are stored in [i, j] or [j, i],
                # so we check, and take transpose if necessary.

                if key_j in self.matrix.get(key_i, {}).keys():
                    values = self.matrix[key_i][key_j]
                elif key_i in self.matrix.get(key_j, {}).keys():
                    values = self.matrix[key_j][key_i].T
                else:
                    values = np.zeros(shape)

                matrix[index_i : index_i + size_i, index_j : index_j + size_j] = values
                index_j += size_j
            index_i += size_i
        return matrix

    def get_matrix_sparse(self, variables=None, output_type="coo", verbose=False):
        """Return a sparse matrix in desired format.

        :param variables: same as in self.get_matrix, but None is not allowed
        """
        if variables:
            if type(variables) in [list, set]:
                try:
                    variable_dict_i = self.generate_variable_dict_i(variables)
                    variable_dict_j = self.generate_variable_dict_j(variables)
                except KeyError:
                    raise TypeError(
                        "When caling get_matrix with a list, all keys of the list have to be present in the matrix. Otherwise, call get_matrix with a dict of the same type as self.variable_dict_i!"
                    )

            elif type(variables) == tuple:
                if type(variables[0]) == list:
                    try:
                        variable_dict_i = self.generate_variable_dict_i(variables[0])
                        variable_dict_j = self.generate_variable_dict_j(variables[1])
                    except KeyError:
                        raise TypeError(
                            "When caling get_matrix with a tuple of lists, all keys of each list have to be present in the matrix. Otherwise, call get_matrix with a dict of the same type as self.variable_dict_i!"
                        )
                elif type(variables[0]) == dict:
                    variable_dict_i = variables[0]
                    variable_dict_j = variables[1]
                else:
                    raise TypeError(
                        "Each element of varaible tuple must be a dict or list."
                    )
            elif type(variables) == dict:
                variable_dict_i = variable_dict_j = variables
            else:
                raise ValueError("variables must be of type dict, tuple or list.")
        else:
            variable_dict_i = self.variable_dict_i
            variable_dict_j = self.variable_dict_j

            # make sure that both i and j have the same order before getting the matrix.
            if self.symmetric:
                if variable_dict_i != variable_dict_j:
                    print("Warning, variable_dict_j not equal to variable_dict_i")
                    variable_dict_i = variable_dict_j

        import time

        t1 = time.time()
        if verbose:
            print(f"Finding nonzero elements took {time.time() - t1:.2}s.")
        t1 = time.time()
        i_list = np.array([], dtype=int)
        j_list = np.array([], dtype=int)
        data_list = np.array([], dtype=float)
        indices_i = generate_indices(variable_dict_i)
        indices_j = generate_indices(variable_dict_j)
        # Loop through blocks of stored matrices
        for key_i in self.matrix:
            for key_j in self.matrix[key_i]:
                # Check if blocks appear in variable dictionary
                if key_i in variable_dict_i and key_j in variable_dict_j:
                    values = self.matrix[key_i][key_j]
                    assert values.shape == (
                        variable_dict_i[key_i],
                        variable_dict_j[key_j],
                    ), f"Variable size does not match input matrix size, variables: {(variable_dict_i[key_i],variable_dict_j[key_j])}, matrix: {values.shape}"
                    # generate list of indices for sparse mat input
                    rows, cols = np.nonzero(values)
                    i_list = np.append(i_list, rows + indices_i[key_i])
                    j_list = np.append(j_list, cols + indices_j[key_j])
                    data_list = np.append(data_list, values[rows, cols])

        if verbose:
            print(f"Filling took {time.time() - t1:.2}s.")

        shape = get_shape(variable_dict_i, variable_dict_j)

        t1 = time.time()
        if output_type == "coo":
            mat = sp.coo_matrix((data_list, (i_list, j_list)), shape=shape)
        elif output_type == "csr":
            mat = sp.csr_matrix((data_list, (i_list, j_list)), shape=shape)
        elif output_type == "csc":
            mat = sp.csc_matrix((data_list, (i_list, j_list)), shape=shape)
        else:
            raise ValueError(f"Unknown matrix type {output_type}")

        if verbose:
            print(f"Filling took {time.time() - t1:.2}s.")

        return mat

    def get_vector(self, variables=None, **kwargs):
        """

        examples:
            vector_dict = dict(x1=4, x2=5, ...)
            f = get_vector(**vector_dict)
            f = get_vector(x1=4, x2=5, ...)
        """
        assert self.symmetric, "get_vector is ambiguous with assymmetric matrices"
        if variables:
            variable_dict = {v: self.variable_dict_i[v] for v in variables}
        else:
            variable_dict = self.variable_dict_i

        vector = np.empty(get_size(variable_dict))
        index = 0
        for key, size in variable_dict.items():
            if key not in kwargs.keys():
                vector[index : index + size] = np.zeros(size)
            else:
                vector[index : index + size] = kwargs[key]
            index += size
        return vector

    # TODO(FD) specific to range-only & LDL implementation. Move to subclass?
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

    # TODO(FD) specific to range-only & LDL implementation. Move to subclass?
    def get_block_matrices(self, key_list=None):
        """Get blocks from PolyMatrix, to be used in block-wise decompositions such as LDL."""
        if key_list is None:
            key_list = list(
                zip(self.variable_dict_i.keys(), self.variable_dict_j.keys())
            )

        # example 1: key_list = [("x1", "x2"), ("x1", "x3"),...]
        # example 2: key_list = [(["x1", "z1"], ["x2", "z2"]), (["x1", "z1"], ["x3", "z3"])...]
        blocks = []
        for key_pair in key_list:
            if (type(key_pair) is tuple) and (type(key_pair[0]) is list):
                key_i, key_j = key_pair
                assert type(key_j) is list, "Mixed types not allowed"
                # example 2: append submatrices
                blocks.append(
                    self.get_matrix(variables=(key_i, key_j), output_type="dense")
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
                    i_size = self.variable_dict_i[key_i]
                    j_size = self.variable_dict_j[key_j]
                    blocks.append(np.zeros((i_size, j_size)))
        return blocks

    def plot_matrix(self, plot_type, variables, **kwargs):
        if variables is None:
            variables_i = self.generate_variable_dict_i()
            if self.symmetric:
                variables_j = self.generate_variable_dict_i()
            else:
                variables_j = self.generate_variable_dict_j()
        else:
            variables_i = self.generate_variable_dict_i(variables)
            if self.symmetric:
                variables_j = self.generate_variable_dict_i(variables)
            else:
                variables_j = self.generate_variable_dict_j(variables)

        mat = self.get_matrix(variables=(variables_i, variables_j))
        if plot_type == "sparse":
            plt.spy(mat, **kwargs)
        elif plot_type == "dense":
            plt.matshow(mat.toarray(), **kwargs)
        else:
            raise ValueError(plot_type)

        for tick_fun, variables in zip(
            [lambda **kwargs: plt.xticks(**kwargs, rotation=90), plt.yticks],
            [variables_j, variables_i],
        ):
            first = 0
            tick_locs = []
            tick_lbls = []
            for var, sz in variables.items():
                tick_locs += [first + i for i in range(sz)]
                tick_lbls += [str(var) + f":{i}" for i in range(sz)]
                first = first + sz
            tick_fun(ticks=tick_locs, labels=tick_lbls, fontsize=10)

    def spy(self, variables: dict() = None, **kwargs):
        self.plot_matrix(plot_type="sparse", variables=variables, **kwargs)

    def matshow(self, variables: dict() = None, **kwargs):
        self.plot_matrix(plot_type="dense", variables=variables, **kwargs)

    def __repr__(self, variables=None, binary=False):
        """Called by the print() function"""
        if self.shape is None:
            self.shape = self.get_shape()

        output = f"Sparse polymatrix of shape {self.shape}\n"
        if self.shape[0] > 100:
            return output

        output += f"Number of nnz: {self.nnz}\n\n"

        if not variables:
            variables_i = self.variable_dict_i.keys()
            variables_j = self.variable_dict_j.keys()
        else:
            variables_i = variables_j = variables

        import pandas

        df = pandas.DataFrame(columns=variables_i, index=variables_j)
        df.update(self.matrix)
        df = df.transpose()
        if (len(df) > 10) or binary:
            df = df.notnull().astype("int")
        else:
            df.fillna(0, inplace=True)
        return output + df.to_string()

    def __add__(self, other, inplace=False):
        # NOTE: This function runs faster if the second matrix has fewer elements than the first.
        #       This is usually the case when using __iadd__ function.
        if inplace:
            res = self
        else:
            res = deepcopy(self)

        if type(other) is PolyMatrix:
            # # add two different polymatrices
            assert res.symmetric == other.symmetric, TypeError(
                "Both matrices must be symmetric or non-symmetric to add."
            )
            # Loop through second matrix elements
            for key_i in other.adjacency_i:
                for key_j in other.adjacency_i[key_i]:
                    # Check if element exists in first matrix
                    if key_i in res.matrix and key_j in res.matrix[key_i]:
                        # Check shapes of matrices to be added
                        assert (
                            res.matrix[key_i][key_j].shape
                            == other.matrix[key_i][key_j].shape
                        ), ValueError(
                            f"Cannot add PolyMatrix element ({key_i},{key_j}) due to shape mismatch."
                        )
                        res.matrix[key_i][key_j] = (
                            res.matrix[key_i][key_j] + other.matrix[key_i][key_j]
                        )
                    else:
                        # If element does not yet exist, add it. Symmetric flag off to avoid double counting
                        res.__setitem__(
                            (key_i, key_j),
                            val=other.matrix[key_i][key_j],
                            symmetric=False,
                        )
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

    def transpose(self):
        res = deepcopy(self)

        matrix_tmp = {}
        for key_i, key_j_list in res.adjacency_i.items():
            for key_j in key_j_list:
                if key_j in matrix_tmp.keys():
                    matrix_tmp[key_j][key_i] = res.matrix[key_i][key_j].T
                else:
                    matrix_tmp[key_j] = {key_i: res.matrix[key_i][key_j].T}
        res.matrix = matrix_tmp

        tmp = deepcopy(res.adjacency_i)
        res.adjacency_i = res.adjacency_j
        res.adjacency_j = tmp

        tmp = deepcopy(res.variable_dict_i)
        res.variable_dict_i = res.variable_dict_j
        res.variable_dict_j = tmp

        return res

    def multiply(self, other_mat):
        """

        a11 a12  @  b11 b12
        a21 a22     b21 b22
        a11*b11 + a12*b21

        """
        output_mat = PolyMatrix(symmetric=False)

        rows = self.adjacency_i.keys()
        cols = other_mat.adjacency_j.keys()
        for key_i in rows:
            for key_j in cols:
                common_elements = set(self.adjacency_i[key_i]).intersection(
                    other_mat.adjacency_j[key_j]
                )
                for key_mul in common_elements:
                    newval = self[key_i, key_mul] @ other_mat[key_mul, key_j]
                    output_mat[key_i, key_j] += newval
        output_mat.shape = None
        return output_mat

    def invert_diagonal(self, inplace=False):
        if inplace:
            res = self
        else:
            res = deepcopy(self)

        for key_i in self.variable_dict_i:
            try:
                new_matrix = self.matrix[key_i][key_i]
            except:
                continue
            res.matrix[key_i][key_i] = np.linalg.inv(new_matrix)
        return res

    def __iadd__(self, other):
        """Overload the += operation"""
        return self.__add__(other, inplace=True)

    def __imul__(self, other):
        """Overload the *= operation"""
        return self.__mul__(other, inplace=True)
