import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNLE


def run_misspec_ma1():
    def MA1(t1, n_obs=100, batch_size=1, random_state=None):
        r"""Generate a sequence of samples from the MA2 model.
        The sequence is a moving average
            x_i = w_i + \theta_1 w_{i-1}
        where w_i are white noise ~ N(0,1).
        Parameters
        ----------
        t1 : float, array_like
        n_obs : int, optional
        batch_size : int, optional
        random_state : RandomState, optional
        """
        # Make inputs 2d arrays for broadcasting with w
        t1 = np.asanyarray(t1).reshape((-1, 1))

        random_state = random_state or np.random

        # i.i.d. sequence ~ N(0,1)
        w = random_state.randn(batch_size, n_obs + 2)
        x = w[:, 2:] + t1 * w[:, 1:-1]
        return x.reshape((batch_size, -1))  # ensure 2D

    def stochastic_volatility(
        w=-0.736, rho=0.9, sigma_v=0.36, n_obs=100, batch_size=1, random_state=None
    ):
        """Sample for a stochastic volatility model.
        specified in Frazier and Drovandi (2021). This is the true Data
        Generating Process for this example.
        Uses a normally distributed shock term.
        Parameters
        ----------
        w : float, optional
        rho : float, optional
        sigma_v : float, optional
        n_obs : int, optional
        batch_size : int, optional
        random_state : RandomState, optional
        Returns
        -------
        y_mat : np.array
        """
        random_state = random_state or np.random

        h_mat = np.zeros((batch_size, n_obs))
        y_mat = np.zeros((batch_size, n_obs))

        w_vec = np.repeat(w, batch_size)
        rho_vec = np.repeat(rho, batch_size)
        sigma_v_vec = np.repeat(sigma_v, batch_size)

        h_mat[:, 0] = w_vec + random_state.normal(0, 1, batch_size) * sigma_v_vec
        y_mat[:, 0] = np.exp(h_mat[:, 0] / 2) * random_state.normal(0, 1, batch_size)

        # TODO! CONFIRM CHANGE TO RANGE
        for i in range(1, n_obs):
            h_mat[:, i] = (
                w_vec
                + rho_vec * h_mat[:, i - 1]
                + random_state.normal(0, 1, batch_size) * sigma_v_vec
            )
            y_mat[:, i] = np.exp(h_mat[:, i] / 2) * random_state.normal(
                0, 1, batch_size
            )

        return y_mat.reshape((batch_size, -1))  # ensure 2d

    def autocov(x, lag=0):
        """Return the autocovariance.
        Assumes a (weak) univariate stationary process with mean 0.
        Realizations are in rows.
        Parameters
        ----------
        x : np.array of size (n, m)
        lag : int, optional
        Returns
        -------
        C : np.array of size (n,)
        """
        x = np.atleast_2d(x)
        # In R this is normalized with x.shape[1]
        if lag == 0:
            C = np.mean(x[:, :] ** 2, axis=1)
        else:
            C = np.mean(x[:, lag:] * x[:, :-lag], axis=1)

        return C

    def summstats(x):
        s1 = autocov(x)
        s2 = autocov(x, lag=1)
        return np.array([s1, s2]).reshape(1, -1)

    def simulation_wrapper(params):
        x_sim = MA1(params)

        sim_sum = torch.as_tensor(summstats(x_sim))
        return sim_sum.reshape((1, -1))

    seed_obs = 1
    true_params = [-0.736, 0.9, 0.36]
    n_obs = 50

    y = stochastic_volatility(
        *true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs)
    )

    prior_min = [-1]
    prior_max = [1]
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )

    posterior = infer(
        simulation_wrapper, prior, method="SNLE", num_simulations=300, num_workers=2
    )

    # posterior =

    y = simulation_wrapper(true_params)
    samples = posterior.sample((10000,), x=y)
    fig, axes = analysis.pairplot(
        samples,
        limits=[[0.5, 80], [1e-4, 15.0]],
        ticks=[[0.5, 80], [1e-4, 15.0]],
        figsize=(5, 5),
        # points=true_params,
        points_offdiag={"markersize": 6},
        points_colors="r",
    )
    plt.savefig("misspec_ma1.png")


if __name__ == "__main__":
    run_misspec_ma1()
