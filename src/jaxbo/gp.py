import jax
import jax.numpy as jnp
from jax.numpy.linalg import inv, slogdet
from jax.scipy.linalg import solve
from jaxopt import GradientDescent
from jaxtyping import Array, Float


@jax.jit
def rbf(x1: Float[Array, " d"], x2: Float[Array, " d"], alpha: float, beta: float):
    d = x1 - x2
    v = alpha * jax.numpy.exp(-d.dot(d) / beta)
    return v


rbf_vector = jax.jit(jax.vmap(rbf, in_axes=(0, None, None, None)))
rbf_matrix = jax.jit(jax.vmap(rbf_vector, in_axes=(None, 0, None, None)))


def _objective(
    params: Float[Array, "2"],
    X: Float[Array, "n d"],
    y: Float[Array, " n"],
):
    # Compute the negative log likelihood
    # constant term is omitted
    alpha, beta, gamma = jnp.exp(params)
    n = y.shape[0]
    K = rbf_matrix(X, X, alpha, beta) + gamma * jnp.identity(n)
    neg_log_lik = y.dot(solve(K, y, assume_a="pos")) + slogdet(K)[1]
    return neg_log_lik / n


class GaussianProcessRegression:
    def __init__(self, alpha: float, beta: float, gamma: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _fit(self, X: Float[Array, "n d"], y: Float[Array, " n"]):
        # Compute the inverse of the covariance matrix
        self.X = X
        n, _d = X.shape
        K = rbf_matrix(X, X, self.alpha, self.beta) + self.gamma * jnp.identity(n)
        self.K_inv_ = inv(K)
        self.K_inv_y_ = self.K_inv_.dot(y)

    def fit(self, X: Float[Array, "n d"], y: Float[Array, " n"], optimize: bool = True):
        if optimize:
            x0 = jnp.log(jnp.array([self.alpha, self.beta, self.gamma]))
            optimizer = GradientDescent(_objective, jit=True, stepsize=1e-5)
            params, _state = optimizer.run(x0, X, y)
            self.alpha, self.beta, self.gamma = jnp.exp(params)
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
