import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import pymc3 as pm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
pd.set_option('display.max_columns', 100, 'display.max_rows', 100)


def get_data():
    # load data
    srrs2 = pd.read_csv(pm.get_data('srrs2.dat'))
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs2['county'] = srrs2['county'].str.strip()
    srrs_mn = srrs2.loc[srrs2.state == 'MN', :].copy()
    srrs_mn['fips'] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips

    cty = pd.read_csv(pm.get_data('cty.dat'))
    cty_mn = cty.loc[cty.st == 'MN', :].copy()
    cty_mn['fips'] = cty_mn.stfips * 1000 + cty_mn.ctfips

    srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')

    # target (y)
    y = srrs_mn['activity'].values
    y_bc, lambda_bc = boxcox(y + 0.1)

    # predictor (x)
    x = srrs_mn[['floor']].values

    # groups
    group = pd.Categorical(srrs_mn['county'])
    group_idx = group.codes

    return srrs_mn, x, y, y_bc, lambda_bc, group, group_idx


class MultiLevelModel(object):
    """
    https://discourse.pymc.io/t/how-do-we-predict-on-new-unseen-groups-in-a-hierarchical-model-in-pymc3/2571/2
    """
    def __init__(self):
        self.trace_ = None
        self.n_groups_ = None
        self.n_features_ = None
        self.model_ = None

    def _build_model(self, X, y, **kwargs):
        raise NotImplementedError()

    def train(self, X, y, draws=4000, tune=2000, chains=4, cores=4,
              target_accept=.95, burn=500, **model_kwargs):

        with self._build_model(X=X, y=y, **model_kwargs) as m:
            self.trace_ = pm.sample(
                draws=draws, tune=tune, chains=chains,
                cores=cores, target_accept=target_accept)[burn:]

        self.model_ = m

        return self

    def predict(self, X, varnames=None, samples=1000, **model_kwargs):
        # dummy y to match shape of X
        y = np.zeros(X.shape[0])

        with self._build_model(X=X, y=y, **model_kwargs) as m:
            # extract the learnt global effect from the train_trace
            df = pm.trace_to_dataframe(self.trace_,
                                       varnames=varnames,
                                       include_transformed=True)

            pred = pm.sample_posterior_predictive(trace=df.to_dict('records'),
                                                  samples=samples)

            return pred


class PoolModel(MultiLevelModel):
    def _build_model(self, X, y, **kwargs):
        with pm.Model() as model:
            # priors
            alpha = pm.Normal('alpha', mu=0, sigma=1e5)
            beta = pm.Normal('beta', mu=0, sigma=1e5)
            sigma = pm.HalfNormal('sigma', sigma=1e5)

            # mean: linear regression
            mu = alpha + pm.math.dot(X, beta)

            # degree of freedom
            nu = pm.Exponential('nu', 1 / 30)

            # observations
            pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)

        return model


class UnpoolModel(MultiLevelModel):
    def _build_model(self, X, y, **kwargs):
        group_idx = kwargs['group_idx']

        if self.n_groups_ is None:
            self.n_groups_ = np.unique(group_idx).shape[0]

        if self.n_features_ is None:
            self.n_features_ = X.shape[-1]

        with pm.Model() as model:
            # priors
            alpha = pm.Normal('alpha', mu=0, sigma=1e5, shape=(self.n_features_, self.n_groups_))
            beta = pm.Normal('beta', mu=0, sigma=1e5, shape=(self.n_features_, self.n_groups_))
            sigma = pm.HalfNormal('sigma', sigma=1e5)

            # mean: linear regression
            mu = alpha[:, group_idx] + pm.math.dot(X, beta[:, group_idx])

            # degree of freedom
            nu = pm.Exponential('nu', 1 / 30)

            # observations
            pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)

        return model


class PartialPoolModel(MultiLevelModel):
    def _build_model(self, X, y, **kwargs):
        group_idx = kwargs['group_idx']
        n_groups = self.n_groups if self.n_groups is not None else np.unique(group_idx).shape[0]
        n_features = X.shape[-1]

        with pm.Model() as model:
            # intercept hyper-priors
            alpha_mu = pm.Normal('alpha_mu', mu=0, sigma=1e5)
            alpha_sigma = pm.HalfNormal('sigma', sigma=1e5)

            # intercept prior
            alpha = pm.Normal('alpha', mu=alpha_mu, sigma=alpha_sigma, shape=(n_features, n_groups))

            # slope hyper-priors
            beta_mu = pm.Normal('beta_mu', mu=0, sigma=1e5)
            beta_sigma = pm.HalfNormal('sigma', sigma=1e5)

            # slope prior
            beta = pm.Normal('beta', mu=beta_mu, sigma=beta_sigma, shape=(n_features, n_groups))

            # model error
            sigma = pm.HalfNormal('sigma', sigma=1e5)

            # mean: linear regression
            mu = alpha[:, group_idx] + pm.math.dot(X, beta[:, group_idx])

            # degree of freedom
            nu = pm.Exponential('nu', lam=1 / 30.)

            # data likelihood
            pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=y)

            ###
            n_groups = county_cat.categories.shape[0]

            # intercept hyper-priors
            alpha_mu = pm.Normal('alpha_mu', mu=0, sigma=100)
            alpha_sigma = pm.HalfCauchy('alpha_sigma', beta=5)

            # intercept prior
            alpha_t = pm.Normal('alpha_t', mu=0, sigma=1, shape=n_groups)
            alpha = pm.Deterministic('alpha', alpha_mu + alpha_sigma * alpha_t)

            # slope hyper-priors
            beta_mu = pm.Normal('beta_mu', mu=0, sigma=100)
            beta_sigma = pm.HalfCauchy('beta_sigma', beta=5)

            # slope prior
            beta_t = pm.Normal('beta_t', mu=0, sigma=1, shape=n_groups)
            beta = pm.Deterministic('beta', beta_mu + beta_sigma * beta_t)

            # model error
            sigma = pm.HalfNormal('sigma', sigma=10 * radon_activity_bc.std())

            # degree of freedom
            nu = pm.Exponential('nu', lam=1 / 30.)

            # expected value
            mu = alpha[county_cat_idx] + beta[county_cat_idx] * floor

            # data likelihood
            y = pm.StudentT('y', mu=mu, sigma=sigma, nu=nu, observed=radon_activity_bc)

        return model


def plot_prediction(x, y, prob=(0.025, 0.5, 0.975), xlabel=None, ylabel=None,
                    ci=True, ax=None, figsize=(18, 6), **kwargs):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    q = mquantiles(y, prob=prob, axis=0)
    ax.plot(x, q[1], **kwargs)
    if ci:
        ax.fill_between(x, q[0], q[2], alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return fig, ax


def cv(func, X, y, n_splits, random_state=None):
    folds = KFold(n_splits=n_splits, random_state=random_state)
    cv_scores = np.empty_like(n_splits)

    for i, (train_index, test_index) in enumerate(folds.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_pred = func()
        cv_scores[i] = np.sqrt(mean_squared_error(y_test, y_pred))

    return cv_scores


if __name__ == '__main__':
    dataset, x, y, y_bc, lambda_bc, group, group_idx = get_data()

    # model = PoolModel()
    # model.train(x, y_bc, draws=3000, tune=2000, chains=4, cores=4, target_accept=.85, burn=500)
    #
    # x_ = np.linspace(x.min(), x.max(), 50)
    # y_pred = model.predict(x_, samples=1000)['y']
    #
    # fig, ax = plt.subplots(figsize=(7, 6))
    # ax.scatter(x, y_bc)
    # plot_prediction(x_, y_pred, xlabel='floor', ylabel='BoxCox(radon activity)', ci=True,
    #                 ax=ax, linestyle='--', c='b', label='pooled')

    model = UnpoolModel()
    model.train(x, y_bc, draws=30, tune=30, chains=1, cores=4, target_accept=.85, burn=0, group_idx=group_idx)
    x_ = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
    y_pred = model.predict(x_, samples=500, group_idx=np.ones(50, dtype=np.int16) * 4)['y']

    idx = range(0, group.categories.shape[0], 10)
    selected_county = group.value_counts().sort_values(ascending=False).iloc[idx].index.tolist()

    fig, ax = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    ax_ = ax.ravel()

    for i, county in enumerate(selected_county):
        # plot observed data points
        mask = dataset['county'] == county
        ax_[i].scatter(x[mask], y_bc[mask])
        j = np.where(group.categories == county)[0][0]
        x = np.linspace(x.min(), x.max(), 50)

        # compare pooled and unpooled models
        # pooled model
        plot_prediction(x, y_pooled, xlabel='floor', ylabel='BoxCox(radon activity)', ci=True, ax=ax_[i],
                        linestyle='--', c='b', label='pooled')

        # unpooled model
        y_unpooled = chain_unpooled['alpha'][:, j] + chain_unpooled['beta'][:, j] * x[:, None]
        plot_prediction(x, y_unpooled, xlabel='floor', ylabel='BoxCox(radon activity)', ci=True, ax=ax_[i],
                        linestyle='-.', c='g', label='unpooled')

    fig.tight_layout()
    plt.show()
    plt.show()
