import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from library.models.nn import *
from library.visualize import *

key = jax.random.key(999)
params = init_dense_neural_network(key, [2, 10, 10, 1])
model = jax.vmap(lambda p,x: dense_neural_network(p, x, ha=jax.nn.tanh)[0,0], in_axes=(None, 0))
loss_fn = lambda p,x,y: loss_mse(model(p, x), y)

data_xor = jnp.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
data_x, data_y = data_xor[:, :2], data_xor[:, 2]
print(data_x)
print(data_y)
print(data_x.shape)
print(data_y.shape)

opt = optax.sgd(1e-1)
opt_state = opt.init(params)

@jax.jit
def opt_step(opt_state_, params_, x, y):
	loss, grad = jax.value_and_grad(loss_fn)(params_, x, y)
	updates, opt_state_ = opt.update(grad, opt_state_, params_)
	params_ = optax.apply_updates(params_, updates)
	return loss, opt_state_, params_

for i in range(500):
	loss, opt_state, params = opt_step(opt_state, params, data_x, data_y)
	if i % 10 == 0:
		print(i, loss)

grid_x, grid_y = jnp.meshgrid(jnp.linspace(0,1,10), jnp.linspace(0,1,10))
grid_z = model(params, jnp.array([grid_x, grid_y]).T.reshape(-1,2)).reshape((10,10))

plt.imshow(grid_z, extent=(0,1,0,1))
_ = plot_surface3d(grid_x, grid_y, grid_z)
plt.show()
