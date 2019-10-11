import numpy as np
import pandas as pd
from scipy.stats import boxcox, t
from scipy.special import inv_boxcox
import pymc3 as pm
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
pd.set_option('display.max_columns', 100, 'display.max_rows', 100)


class MultiLevelModel(object):
    """
    Base class for a multi-level model
    """

    def __init__(self):
        self.trace_ = None
        self.n_groups_ = None
        self.n_features_ = None
        self.model_ = None

    def _build_model(self, X, y, **kwargs):
        raise NotImplementedError()

    def fit(self, X, y, draws=4000, tune=2000, chains=4, cores=4,
            target_accept=.8, burn=500, **model_kwargs):

        self.model_ = self._build_model(X=X, y=y, **model_kwargs)

        with self.model_:
            self.trace_ = pm.sample(
                draws=draws, tune=tune, chains=chains,
                cores=cores, target_accept=target_accept)[burn:]

        return self

    def predict(self, X, **kwargs):
        raise NotImplementedError()

    def cv(self, X, y, n_splits, random_state=None, **obj_func_kwargs):
        folds = KFold(n_splits=n_splits, random_state=random_state)
        cv_scores_train = np.empty(n_splits)
        cv_scores_test = np.empty(n_splits)

        for i, (train_idx, test_idx) in enumerate(folds.split(X)):

            cv_scores_train[i], cv_scores_test[i] = \
                self._objective_func(X, y, train_idx, test_idx, **obj_func_kwargs)

        return cv_scores_train.mean(), cv_scores_train.std(), \
               cv_scores_test.mean(), cv_scores_test.std()

    def _objective_func(self, X_train, y_train, X_test, y_test, **kwargs):
        raise NotImplementedError()


class PoolLinearModel(MultiLevelModel):
    def _build_model(self, X, y, **kwargs):
        with pm.Model() as model:
            # priors
            alpha = pm.Normal('alpha', mu=0, sigma=1e5)
            beta = pm.Normal('beta', mu=0, sigma=1e5)
            sigma = pm.HalfNormal('sigma', sigma=1e5)

            # mean: linear regression
            mu = alpha + beta * X  # alpha + pm.math.dot(beta, X)

            # degree of freedom
            nu = pm.Exponential('nu', 1 / 30)

            # observations
            pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)

        return model

    def predict(self, X, **kwargs):
        mu = self.trace_['alpha'] + self.trace_['beta'] * X[:, None]
        dist = t(df=self.trace_['nu'], loc=mu, scale=self.trace_['sigma'])
        if kwargs.get('q') is None:
            return dist, dist.mean().mean(axis=1)
        else:
            return dist, [dist.ppf(q_).mean(axis=1) for q_ in kwargs['q']]

    def _objective_func(self, X_train, y_train, X_test, y_test, **kwargs):
        self.fit(X_train, y_train, **kwargs['fit'])

        _, y_pred_train = self.predict(X_train)
        _, y_pred_test = self.predict(X_test)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        return rmse_train, rmse_test


class PartialPoolLinearModel(MultiLevelModel):
    def _build_model(self, X, y, **kwargs):
        group_idx = kwargs['group_idx']
        n_groups = np.unique(group_idx).shape[0]
        # n_features = X.shape[0]

        with pm.Model() as model:
            # intercept hyper-priors
            alpha_mu = pm.Normal('alpha_mu', mu=0, sigma=1e5)
            alpha_sigma = pm.HalfNormal('alpha_sigma', sigma=1e5)

            # intercept prior
            alpha_t = pm.Normal('alpha_t', mu=0, sigma=1, shape=n_groups)
            alpha = pm.Deterministic('alpha', alpha_mu + alpha_sigma * alpha_t)

            # slope hyper-priors
            beta_mu = pm.Normal('beta_mu', mu=0, sigma=1e5)
            beta_sigma = pm.HalfNormal('beta_sigma', sigma=1e5)

            # slope prior
            beta_t = pm.Normal('beta_t', mu=0, sigma=1, shape=n_groups)
            beta = pm.Deterministic('beta', beta_mu + beta_sigma * beta_t)

            # model error
            sigma = pm.HalfNormal('sigma', sigma=1e5)

            # degree of freedom
            nu = pm.Exponential('nu', lam=1 / 30.)

            # expected value
            mu = alpha[group_idx] + beta[group_idx] * X

            # data likelihood
            pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)

        return model

    def predict(self, X, **kwargs):
        group_idx = kwargs['group_idx']
        mu = self.trace_['alpha'][:, group_idx] + self.trace_['beta'][:, group_idx] * X[:, None]
        dist = t(df=self.trace_['nu'], loc=mu, scale=self.trace_['sigma'])
        if kwargs.get('q') is None:
            return dist, dist.mean().mean(axis=1)
        else:
            return dist, [dist.ppf(q_).mean(axis=1) for q_ in kwargs['q']]

    def _objective_func(self, X, y, train_idx, test_idx, **kwargs):
        # TODO: make sure group index are assigned properly
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        kwargs_ = copy.copy(kwargs)
        kwargs_['fit']['group_idx'] = kwargs['fit']['group_idx'][train_idx]
        self.fit(X_train, y_train, **kwargs_['fit'])

        _, y_pred_train = self.predict(X_train, **kwargs['predict'])
        _, y_pred_test = self.predict(X_test, **kwargs['predict'])

        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        return rmse_train, rmse_test


class UnpoolLinearModel(PartialPoolLinearModel):
    def _build_model(self, X, y, **kwargs):
        group_idx = kwargs['group_idx']
        n_groups = np.unique(group_idx).shape[0]
        # n_features = X.shape[0]

        with pm.Model() as model:
            # priors
            alpha = pm.Normal('alpha', mu=0, sigma=1e5, shape=n_groups)  # shape=(n_groups, n_features)
            beta = pm.Normal('beta', mu=0, sigma=1e5, shape=n_groups)  # shape=(n_groups, n_features)
            sigma = pm.HalfNormal('sigma', sigma=1e5)

            # mean: linear regression
            mu = alpha[group_idx] + beta[group_idx] * X  # alpha[group_idx] + pm.math.dot(beta[group_idx], X)

            # degree of freedom
            nu = pm.Exponential('nu', 1 / 30.)

            # observations
            pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)

        return model


def plot_prediction(x, y_mean, y_upper=None, y_lower=None, xlabel=None,
                    ylabel=None, ax=None, figsize=(18, 6), **kwargs):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    ax.plot(x, y_mean, **kwargs)
    if not (y_upper is None or y_lower is None):
        ax.fill_between(x, y_upper, y_lower, alpha=kwargs.get('alpha', 0.5))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return fig, ax
