import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np

caches = {}
for k in glob.glob("H_CACHE_*"):
	with open(k, 'rb') as f:
		caches.update({k:pickle.load(f)})
print(caches.keys())

s_tanh = caches['H_CACHE_squashed_tanh_250811.pkl']
a_axis = s_tanh['b_axis']
s_tanh_loss = [s_tanh['results'][i]['test_loss'][-1].item() for i in range(len(a_axis))]
opt_s_tanh = np.argmin(s_tanh_loss)
print(s_tanh_loss)

stan = caches['H_CACHE_stan_250811.pkl']
b_axis = stan['b_axis']
stan_loss = [stan['results'][i]['test_loss'][-1].item() for i in range(len(b_axis))]
opt_stan = np.argmin(stan_loss)
print(stan_loss)

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

ax.plot(a_axis, [0]*len(a_axis), s_tanh_loss, c='darkgreen')
ax.plot([1]*len(b_axis), b_axis, stan_loss, c='darkgreen')
ax.set_xlabel('a')
ax.set_xticks(a_axis, [f"{x:.1f}" for x in a_axis])
ax.set_ylabel('b')
ax.set_yticks(b_axis, [f"{x:.1f}" for x in b_axis])
ax.set_zlabel('Test loss')

ax.scatter([(a_axis[0])], [(b_axis[0])], [(s_tanh_loss[0])], marker='x', s=32, c='red', label=f"Tanh a=1, b=0, test_loss={s_tanh_loss[0]:.4f}")
ax.scatter([(a_axis[opt_s_tanh])], [(b_axis[0])], [(s_tanh_loss[opt_s_tanh])],
	marker='x', s=32, c='blue', label=f"Tanh a={a_axis[opt_s_tanh]:.1f}, b=0, test_loss={s_tanh_loss[opt_s_tanh]:.4f}"
)
ax.scatter([(a_axis[0])], [(b_axis[opt_stan])], [(stan_loss[opt_stan])],
	marker='x', s=32, c='orange', label=f"Stan a=1, b={b_axis[opt_stan]:.1f}, test_loss={stan_loss[opt_stan]:.4f}"
)
ax.legend()
ax.view_init(elev=25, azim=45)

plt.show()
