import jax
import jax.numpy as jnp
from jax.numpy import linalg
from jax.scipy.linalg import solve
from jax.scipy.optimize import minimize
from jaxtyping import Array, Float


@jax.jit
def rbf(x1: Float[Array, " d"], x2: Float[Array, " d"], alpha: float, beta: float):
    d = x1 - x2
    v = alpha * jax.numpy.exp(-d.dot(d) / beta)
    return v


rbf_vector = jax.jit(jax.vmap(rbf, in_axes=(0, None, None, None)))
rbf_matrix = jax.jit(jax.vmap(rbf_vector, in_axes=(None, 0, None, None)))


def _log_likelihood(
    alpha: float,
    beta: float,
    gamma: float,
    X: Float[Array, "n d"],
    y: Float[Array, " n"],
):
    n = y.shape[0]
    K = rbf_matrix(X, X, alpha, beta) + gamma * jax.numpy.eye(n)
    return -0.5 * (
        y.dot(solve(K, y, assume_a="pos"))
        + linalg.slogdet(K)[1]
        + n * jax.numpy.log(2 * jax.numpy.pi)
    )


class GaussianProcessRegression:
    def __init__(self, alpha: float, beta: float, gamma: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _fit(self, X: Float[Array, "n d"], y: Float[Array, " n"]):
        # Compute the inverse of the covariance matrix
        self.X = X
        n, _d = X.shape
        K = rbf_matrix(X, X, self.alpha, self.beta) + self.gamma * jax.numpy.eye(n)
        self.K_inv_ = linalg.inv(K)
        self.K_inv_y_ = self.K_inv_.dot(y)

    def fit(self, X: Float[Array, "n d"], y: Float[Array, " n"], optimize: bool = True):
        if optimize:
            x0 = jnp.array([self.alpha, self.beta, self.gamma])
            res = minimize(
                lambda x: -_log_likelihood(x[0], x[1], x[2], X, y),
                x0=x0,
                method="BFGS",
            )
            self.alpha, self.beta, self.gamma = res.x
        self._fit(X, y)

    def predict_one(self, x: Float[Array, " d"]):
        kstar = rbf_vector(self.X, x, self.alpha, self.beta)
        mu = kstar.dot(self.K_inv_y_)
        sigma2 = (
            rbf(x, x, self.alpha, self.beta)
            + self.gamma
            - kstar.dot(self.K_inv_.dot(kstar))
        )
        return mu, sigma2
