#entropy measures the uncertainty of a function
def entropy(y):
    hist = np.bincount(y) #this will count number of occ of all class labels
    ps = hist / len(y)

    return -np.sum([p* np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value = None):
        self.feature=feature
        self.threshold=threshold,
        self.left=left
        self.right=right
        self.value=value

    def is_leaf_node(self):
        return self.value is not None
    

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

 
    def fit(self, X, y):
        #grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root = self._grow_tree(X,y)


    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        #stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        
        #greedy search
        best_feature, best_thresh = self._best_criteria(X,y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:,best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1)

        return Node(best_feature, best_thresh, left, right)


    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0] #occ of 1, first tuple in list, tuple has value and occ and we want the value so (1)[0][0]
        return most_common
    
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx,split_threshold
    
    def _information_gain(self, y, X_col, split_thresh):
        #parent E
        parent_entropy = entropy(y)

        #generate split
        left_idxs, right_idxs = self._split(X_col, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        #weighted avg child E
        n = len(y)
        n_l, n_r = len(left_idxs),len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        #return information gain
        ig = parent_entropy - child_entropy
        return ig


    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten() #flatten() makes it 1D vector
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs


    def predict(self, X):
        return np.array([self._traverse_tree(x,self.root)for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)










import numpy as np
#from ..DecisionTree.main import DecisionTree
from collections import Counter

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]

class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.tress = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)


    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        #majority vote
        #from [1111 0000 1111]
        #to [101 101 101 101]
        tree_preds = np.swapaxes(tree_preds, 0,1)
        
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    
    def most_common_label(self, y):
        counter = Counter()
        most_common = counter.most_common(1)[0][0]
        return most_common
    


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def accuracy(y_pred, y_true):
    return np.sum(y_pred==y_true)/len(y_true)

if __name__ == '__main__':

    data = load_breast_cancer
    X = data.data
    y = data.target

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    rf = RandomForest(n_trees=3)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print('accurage is ', acc)