import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.base import infer


def run_MA2():
    def MA2(params, n_obs=50, batch_size=1, random_state=None):
        r"""Generate a sequence of samples from the MA2 model.

        The sequence is a moving average

            x_i = w_i + \theta_1 w_{i-1} + \theta_2 w_{i-2}

        where w_i are white noise ~ N(0,1).

        Parameters
        ----------
        t1 : float, array_like
        t2 : float, array_like
        n_obs : int, optional
        batch_size : int, optional
        random_state : RandomState, optional

        """
        # Make inputs 2d arrays for broadcasting with w
        if isinstance(params, torch.Tensor):
            t1, t2 = float(params[0][0]), float(params[0][1])
        else:
            t1, t2 = params[0], params[1]
        random_state = random_state or np.random

        # i.i.d. sequence ~ N(0,1)
        w = random_state.randn(batch_size, n_obs + 2)
        x = w[:, 2:] + t1 * w[:, 1:-1] + t2 * w[:, :-2]
        return x

    def simulation_wrapper(params):
        x_sim = MA2(params)
        sim_sum = torch.as_tensor(x_sim.astype("float32"))
        return sim_sum.reshape((-1, 50))  # TODO: magic number 100

    # TODO: ACTUAL PRIOR?
    prior_min = [-1, -1]
    prior_max = [1, 1]
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
    )

    inference = SNLE(prior=prior)

    true_params = np.array([0.6, 0.2])
    y = MA2(true_params)

    num_rounds = 30

    posteriors = []
    proposal = prior

    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(simulation_wrapper, proposal, num_simulations=500)
        density_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(y)

    samples = posterior.sample((5000,), x=y)

    fig, axes = analysis.pairplot(
        samples,
        limits=[[-1, 1], [-1, 1]],
        #    ticks=[[.5, 1], [.5, 15.]],
        figsize=(5, 5),
        #    points=true_params,
        points_offdiag={"markersize": 6},
        points_colors="r",
    )

    plt.savefig("ma2_posterior.png")


if __name__ == "__main__":
    run_MA2()
