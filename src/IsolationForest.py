import numpy as np


def c(n):
    if n > 2:
        return 2 * (np.log(n - 1) + np.euler_gamma) - (2 * (n - 1) / n)
    elif n == 2:
        return 1
    else:
        return 0


class Node:
    def __init__(self, left, right, split_attribute, split_value, size=None):
        self.left = left
        self.right = right
        self.split_attribute = split_attribute 
        self.split_value = split_value
        self.size = size

    def __repr__(self) -> str:
        return str(self.split_attribute) + ' ' + str(self.split_value)

    def is_external(self):
        if self.left == None and self.right == None:
            return True
        else: 
            return False


class ITree:

    def __init__(self, X, current_height, max_height):
        self.tree = self.fit(X, current_height, max_height)

    def fit(self, X, current_height, max_height):
        if current_height >= max_height or len(X) <= 1:
            return Node(None, None, None, None, len(X))
        else:
            split_attribute = np.random.randint(0, X.shape[1])
            col = X[:, split_attribute]

            split_value = float(np.random.uniform(col.min(), col.max()))

            X_l = X[col < split_value, :]
            X_r = X[col >= split_value, :]

            return Node(ITree(X_l, current_height + 1, max_height).tree,
                        ITree(X_r, current_height + 1, max_height).tree,
                        split_attribute,
                        split_value)

    def path_length(self, x, node=None, current_length=0):
        if node == None:
            node = self.tree

        if node.is_external():
            return current_length + c(node.size)

        if x[node.split_attribute] < node.split_value:
            return self.path_length(x, node.left, current_length + 1)
        else:
            return self.path_length(x, node.right, current_length + 1)


class IsolationForest:

    def __init__(self, n_trees, sample_size):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_height = np.ceil(np.log2(sample_size))
        self.forest = []

    def fit(self, X):
        for i in range(self.n_trees):
            sample = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]
            tree = ITree(sample, 0, self.max_height)
            self.forest.append(tree)

    def anomaly_score(self, x):
        expected_path_length = np.mean([tree.path_length(x) for tree in self.forest])
        return 2 ** (- expected_path_length / c(self.sample_size))

    def predict(self, X):
        return np.array([self.anomaly_score(x) for x in X])
