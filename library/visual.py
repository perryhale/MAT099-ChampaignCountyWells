import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_hydrology(
		h_time, # np.ndarray
		k=None, # np.ndarray
		grid_extent=None, # Tuple[float, float, float, float]|None #"(left, right, bottom, top)"
		scatter_data=None, # np.ndarray|None
		scatter_labels=False, # bool
		xlabel=None, # str|None
		ylabel=None, # str|None
		axis_ticks=False, # bool
		cbar = False, # bool
		cbar_label=None, # str|None
		contour_levels=10, # int
		cmap_k = 'BrBG', # str|?~plt.cm.*
		cmap_contour = 'Oranges', # str|?~plt.cm.*
		frame_interval=10, # int
		frame_fps=60, # int
		frame_skip=0, # int
		save_path=None # str|None
	):
	
	# init plot
	fig, ax = plt.subplots(figsize=(8,8))
	if cbar and (k is not None):
		k_colbar = fig.colorbar(ax.imshow(k, cmap=cmap_k), label=cbar_label, fraction=0.03, pad=0.03)
	
	# define frame function
	# type: (int) -> None|List[matplotlib.artist.Artist]
	def update(t):
		
		# skip frames
		t += frame_skip
		
		# setup axis
		ax.clear()
		ax.set_title(f"t={t}")
		if not axis_ticks:
			ax.set_xticks([],[])
			ax.set_yticks([],[])
		if xlabel is not None:
			ax.set_xlabel(xlabel)
		if ylabel is not None:
			ax.set_ylabel(ylabel)
		
		# draw h
		h_contour = ax.contour(h_time[t], levels=contour_levels, cmap=cmap_contour, extent=grid_extent, origin='upper' if (grid_extent is not None) else None)
		h_contour_labels = ax.clabel(h_contour, inline=True, fontsize=8, colors='red')
		#for label in h_contour_labels:
		#	label.set_bbox(dict(facecolor='white', edgecolor='white', boxstyle='square,pad=0.1'))
		
		# draw k
		if k is not None:
			k_im = ax.imshow(k, cmap=cmap_k, extent=grid_extent)
		
		# draw scatter
		if scatter_data is not None:
			scatter_x, scatter_y = scatter_data
			ax.scatter(scatter_x, scatter_y, c='red', s=16, marker='x')
			if scatter_labels:
				for x, y in scatter_data.T:
					ax.text(x, y-0.015, f'({x:.2f}, {y:.2f})', color='red', fontsize=8, ha='center', zorder=999)
		
		###! debug
		#plt.savefig(f"{t}.png")
	
	# render animation
	ani = animation.FuncAnimation(fig, update, frames=len(h_time)-frame_skip, interval=frame_interval)
	
	# output animation
	if save_path is not None:
		ani.save(
			save_path, 
			writer='ffmpeg', 
			fps=frame_fps, 
			progress_callback=lambda i, n: print(f"\33[2K\r[Frame: {i}/{n} ({100*i/n:.2f}%)]", end='')
		)
		plt.close()
		print(f"\nSaved \"{save_path}\"")
	else:
		plt.show()
	
	return None

# type: (jnp.array, jnp.array, jnp.array, jnp.array, tuple[int, int]) -> tuple[Figure, Axes]
def plot_surface3d(grid_x, grid_y, grid_z, k=None, figsize=(5,5), sfc_cmap='prism', cnt_cmap='Oranges', xlabel='X_EPSG_6350', ylabel='Y_EPSG_6350', zlabel='Height (metres)'):
	
	# init figure
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111, projection='3d')
	
	# plot surface and contours
	ax.plot_surface(grid_x, grid_y, grid_z, cmap=sfc_cmap, alpha=0.8, linewidth=0)
	contours = ax.contour(
		grid_x, grid_y, grid_z,
		zdir='z',
		offset=np.min(grid_z)-0.1*(np.max(grid_z)-np.min(grid_z)),
		levels=10,
		cmap=cnt_cmap
	)
	if k is not None:
		ax.plot_surface(grid_x, grid_y, np.ones(grid_z.shape)*np.min(grid_z)-2, facecolors=plt.cm.BrBG(k/k.max()), shade=False)
	
	ax.view_init(elev=20, azim=270)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	ax.set_xticks([],[])
	ax.set_yticks([],[])
	
	return fig, ax

###! debug and styling
# colors='white', linestyles='dashed')
# cmap_trunc = lambda cmap,low,high,levels: plt.cm.colors.LinearSegmentedColormap.from_list('', cmap(np.linspace(low, high, levels)))
# trunc_blues = cmap_trunc(plt.cm.Blues, 0.0, 0.5, contour_levels)
# trunc_reds = cmap_trunc(plt.cm.Reds, 0.5, 1.0, contour_levels)
# k_extent = [grid_x[0,0] - (grid_x[0,1] - grid_x[0,0])/2, grid_x[0,-1] + (grid_x[0,1] - grid_x[0,0])/2, grid_y[0,0] - (grid_y[1,0] - grid_y[0,0])/2, grid_y[-1,0] + (grid_y[1,0] - grid_y[0,0])/2]
# fig, ax = plt.subplots(figsize=(8,8)) if frame_square else plt.subplots(figsize=(8,8*(grid_y.max() - grid_y.min()) / (grid_x.max() - grid_x.min())))
# ax.set_aspect('equal') ###! patch
