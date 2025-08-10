import jax
import jax.numpy as jnp
from jax.nn import tanh
import matplotlib.pyplot as plt

X_MIN = -2
X_MAX = 2
X_RES = 100

B_MIN = 0
B_MAX = 1.5
B_RES = 4

a_fn = lambda b,x: tanh(x) + b * x * tanh(x)
b_cols = plt.cm.Dark2(jnp.linspace(0, 1, B_RES))
bs = jnp.linspace(B_MIN, B_MAX, B_RES)
xs = jnp.linspace(X_MIN, X_MAX, X_RES)

bys = [a_fn(b, xs) for b in bs]
bgys = [jax.vmap(jax.grad(lambda x: a_fn(b, x)))(xs) for b in bs]

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

for b,ys,gys,c in zip(bs, bys, bgys, b_cols):
	ax0.plot(xs, ys, c=c, label=f"β = {b:.1f}")
	ax1.plot(xs, gys, c=c, label=f"β = {b:.1f}")

ax0.axhline(0, linestyle='dashed', c='black')
ax0.axvline(0, linestyle='dashed', c='black')
ax0.set_xlim(X_MIN, X_MAX)
ax0.set_ylabel("Stan(x)")
ax0.set_xlabel("x")
ax0.grid()
ax0.legend()

ax1.axhline(0, linestyle='dashed', c='black')
ax1.axvline(0, linestyle='dashed', c='black')
ax1.set_xlim(X_MIN, X_MAX)
ax1.set_ylabel("δStan(x) / δx")
ax1.set_xlabel("x")
ax1.grid()
ax1.legend()

plt.tight_layout()
plt.show()
