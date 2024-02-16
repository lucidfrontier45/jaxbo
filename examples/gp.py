import jax
import jax.numpy as jnp
import jax.random as jrand
from jaxbo.gp import GaussianProcessRegression


def f(x):
    # y = a||x - x0||^2 + b
    a = 0.2
    b = 3.5
    x0 = jnp.array([1.0, -1.5])
    d = x - x0
    return a * (d * d).sum() + b


n = 100
key = jrand.PRNGKey(0)
key, subkey = jrand.split(key)
X = jrand.normal(subkey, shape=(n, 2)) * 2.5
key, subkey = jrand.split(key)
y = jax.vmap(f)(X) + jrand.normal(subkey, shape=(n,)) * 1.0

gpr = GaussianProcessRegression(alpha=1.0, beta=1.0, gamma=1e-2)
gpr.fit(X, y, optimize=False)
print(gpr.alpha, gpr.beta, gpr.gamma)

x = jnp.array([0.0, 0.0])
mu, sigma2 = gpr.predict_one(x)
correct_y = f(x)

print(correct_y, mu, sigma2)
