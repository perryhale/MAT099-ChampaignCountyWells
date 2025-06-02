import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray|None, 
#	Tuple[float, float, float, float],
#	np.ndarray|None, str|None, str|None, str|None, bool, int, int, int, int, str|None) -> None
def animate_hydrology(grid_x, grid_y, h_time,
		k=None,
		k_extent=None,
		scatter_data=None,
		xlabel=None,
		ylabel=None,
		cbar_label=None,
		no_ticks=False,
		skip_frames=0,
		frame_interval=10,
		frame_fps=60,
		contour_levels=10,
		save_path=None
	):
	
	# unpack
	if scatter_data is not None:
		scatter_x, scatter_y = scatter_data
	
	# init plot
	fig, ax = plt.subplots(figsize=(8,8))# * (grid_y.max() - grid_y.min()) / (grid_x.max() - grid_x.min())))
	
	# setup cmaps, style to be consolidated
	cmap_trunc = lambda cmap,low,high,levels: plt.cm.colors.LinearSegmentedColormap.from_list('', cmap(np.linspace(low, high, levels)))
	trunc_blues = cmap_trunc(plt.cm.Blues, 0.0, 0.5, contour_levels)
	trunc_reds = cmap_trunc(plt.cm.Reds, 0.5, 1.0, contour_levels)
	k_cmap = 'viridis'
	
	# create colorbar
	if k is not None:
		k_cb = fig.colorbar(ax.imshow(k), label=cbar_label, fraction=0.03, pad=0.03)
	
	if k_extent is None:
		# dx = grid_x[0,1] - grid_x[0,0]
		# dy = grid_y[1,0] - grid_y[0,0]
		# x0 = grid_x[0,0] - (grid_x[0,1] - grid_x[0,0])/2
		# x1 = grid_x[0,-1] + (grid_x[0,1] - grid_x[0,0])/2
		# y0 = grid_y[0,0] - (grid_y[1,0] - grid_y[0,0])/2
		# y1 = grid_y[-1,0] + (grid_y[1,0] - grid_y[0,0])/2
		# k_extent = [x0, x1, y0, y1]
		k_extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
	
	# define frame function
	# type: (int) -> None ###! List[matplotlib.artist.Artist]
	def update(t):
		
		# skip frames
		t += skip_frames
		
		# reset axis
		ax.clear()
		#ax.set_aspect('equal')
		ax.set_title(f"t={t}")
		
		# draw surface
		#cf = ax.contourf(grid_x, grid_y, h_time[t], levels=contour_levels, cmap=trunc_reds, alpha=0.5)
		c = ax.contour(grid_x, grid_y, h_time[t], levels=contour_levels, cmap=trunc_blues)#colors='white', linestyles='dashed')
		cl = ax.clabel(c, inline=True, fontsize=8)
		
		# draw optionals
		if k is not None:
			k_im = ax.imshow(k, extent=k_extent, cmap=k_cmap)
		if scatter_data is not None:
			ax.scatter(scatter_x, scatter_y, c='red', s=16, marker='x')
		if xlabel is not None:
			ax.set_xlabel(xlabel)
		if ylabel is not None:
			ax.set_ylabel(ylabel)
		if no_ticks:
			ax.set_xticks([],[])
			ax.set_yticks([],[])
	
	# render animation
	ani = animation.FuncAnimation(fig, update, frames=len(h_time)-skip_frames, interval=frame_interval)
	
	# output animation
	if save_path is not None:
		ani.save(
			save_path, 
			writer='ffmpeg', 
			fps=frame_fps, 
			progress_callback=lambda i, n: print(f"\33[2K\r[Frame: {i}/{n} ({100*i/n:.2f}%)]", end='')
		)
		print(f"\nSaved \"{save_path}\"")
	else:
		plt.show()
