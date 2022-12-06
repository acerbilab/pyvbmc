import numpy as np
import scipy.stats as scs

from pyvbmc import VBMC

np.random.seed(42)


D = 2  # still in 2-D


def log_likelihood(theta):
    """D-dimensional Rosenbrock's banana function."""
    theta = np.atleast_2d(theta)

    x, y = theta[:, :-1], theta[:, 1:]
    return -np.sum((x**2 - y) ** 2 + (x - 1) ** 2 / 100, axis=1)


prior_tau = 3 * np.ones((1, D))  # Length scale of the exponential prior


def log_prior(x):
    """Independent exponential prior."""
    return np.sum(scs.expon.logpdf(x, scale=prior_tau))


def log_joint(x):
    """log-density of the joint distribution."""
    return log_likelihood(x) + log_prior(x)


LB = np.zeros((1, D))  # Lower bounds


UB = 10 * prior_tau  # Upper bounds


PLB = scs.expon.ppf(0.159, scale=prior_tau)
PUB = scs.expon.ppf(0.841, scale=prior_tau)


x0 = np.ones((1, D))  # Optimum of the Rosenbrock function


# initialize VBMC
options = {"plot": True}
vbmc = VBMC(log_joint, x0, LB, UB, PLB, PUB, options)
# Printing will give a summary of the vbmc object:
# bounds, user options, etc.:
print(vbmc)


# run VBMC
vp, results = vbmc.optimize()


lml_true = -1.836  # ground truth, which we know for this toy scenario

print("The true log model evidence is:", lml_true)
print("The obtained ELBO is:", format(results["elbo"], ".3f"))
print("The obtained ELBO_SD is:", format(results["elbo_sd"], ".3f"))


print(vp)


# Generate samples from the variational posterior
n_samples = int(3e5)
Xs, _ = vp.sample(n_samples)

# We compute the pdf of the approximate posterior on a 2-D grid
plot_lb = np.zeros(2)
plot_ub = np.quantile(Xs, 0.999, axis=0)
x1 = np.linspace(plot_lb[0], plot_ub[0], 400)
x2 = np.linspace(plot_lb[1], plot_ub[1], 400)

xa, xb = np.meshgrid(x1, x2)  # Build the grid
xx = np.vstack(
    (xa.ravel(), xb.ravel())
).T  # Convert grids to a vertical array of 2-D points
yy = vp.pdf(xx)  # Compute PDF values on specified points


# You may need to run "jupyter labextension install jupyterlab-plotly" for plotly
# Plot approximate posterior pdf (this interactive plot does not work in higher D)
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Surface(z=yy.reshape(x1.size, x2.size), x=xa, y=xb, showscale=False)
    ]
)
fig.update_layout(
    autosize=False,
    width=800,
    height=500,
    scene=dict(
        xaxis_title="x_0",
        yaxis_title="x_1",
        zaxis_title="Approximate posterior pdf",
    ),
)

# Compute and plot approximate posterior mean
post_mean = np.mean(Xs, axis=0)
fig.add_trace(
    go.Scatter3d(
        x=[post_mean[0]],
        y=[post_mean[1]],
        z=vp.pdf(post_mean),
        name="Approximate posterior mean",
        marker_symbol="diamond",
    )
)

# Find and plot approximate posterior mode
# NB: We do not particularly recommend to use the posterior mode as parameter
# estimate (also known as maximum-a-posterior or MAP estimate), because it
# tends to be more brittle (especially in the PyVBMC approximation) than the
# posterior mean.
post_mode = vp.mode()
fig.add_trace(
    go.Scatter3d(
        x=[post_mode[0]],
        y=[post_mode[1]],
        z=vp.pdf(post_mode),
        name="Approximate posterior mode",
        marker_symbol="x",
    )
)


# # Display posterior also as a non-interactive static image
# from IPython.display import Image

# Image(fig.to_image(format="png"))
