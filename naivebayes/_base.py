import numpy as np

class GaussianNaiveBayes:
    """
    Naive Bayes where P(x_j | y=c) is Gaussian with class-specific mean/var.
    Uses log-probabilities for numerical stability.
    """
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_log_ = None
        self.theta_ = None  # means per class-feature
        self.var_ = None    # variances per class-feature

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.var_   = np.zeros((n_classes, n_features))
        self.class_prior_log_ = np.log(counts / counts.sum())

        # Global variance for smoothing (as in scikit's var_smoothing idea)
        eps = self.var_smoothing * X.var(axis=0).max()

        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[idx] = Xc.mean(axis=0)
            # unbiased variance with smoothing to avoid zero
            self.var_[idx] = Xc.var(axis=0) + eps
        return self

    def _joint_log_likelihood(self, X):
        # log P(y) + sum_j log N(x_j | mu_cj, var_cj)
        X = np.asarray(X, dtype=float)
        n_classes = self.classes_.shape[0]
        jll = np.zeros((X.shape[0], n_classes))
        for idx in range(n_classes):
            mean = self.theta_[idx]
            var = self.var_[idx]
            # log of Gaussian density per feature
            log_prob = -0.5 * (np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var)
            jll[:, idx] = self.class_prior_log_[idx] + log_prob.sum(axis=1)
        return jll

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        # log-softmax -> probabilities
        jll_max = jll.max(axis=1, keepdims=True)
        exp = np.exp(jll - jll_max)
        proba = exp / exp.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]


class MultinomialNaiveBayes:
    """
    Naive Bayes for count features (e.g., word counts).
    Uses Laplace smoothing (alpha) and log probabilities.
    """
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.classes_ = None
        self.class_count_ = None
        self.feature_count_ = None
        self.feature_log_prob_ = None
        self.class_log_prior_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if (X < 0).any():
            raise ValueError("Multinomial NB requires nonnegative feature counts.")
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_count_ = class_counts.astype(float)
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())

        self.feature_count_ = np.zeros((n_classes, n_features), dtype=float)
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.feature_count_[i] = Xc.sum(axis=0)

        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        return self.class_log_prior_ + X @ self.feature_log_prob_.T

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        jll_max = jll.max(axis=1, keepdims=True)
        exp = np.exp(jll - jll_max)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]



