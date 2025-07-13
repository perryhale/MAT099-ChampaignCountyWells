import re
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from library.models.nn import sample_3d_model
from library.visual import animate_hydrology


def resolve_coord_xyt(string):
	try:
		return [tuple(map(float, xyt)) for xyt in re.findall(r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+?)\s*\)', string)]
	except Exception:
		return []

def resolve_vocab_match(words, vocab):
	return all([any([w in v for w in words]) for v in vocab])


def expert_system(h_param, h_fn, k=None, translate=[(0,1)]*4, trend_res=250):
	
	print("+======++======++======++======++======+")
	print("||Champaign County Wells Expert System||")
	print("+======++======++======++======++======+")
	print()
	print(f"Hello my name is WES Champaign, I'm an expert system intended to answer your questions about Champaign's water table. How can I help today?")
	#print("Try asking me a question. If you're unfamiliar with the actions possible within this portal, you can ask for help at any time.")
	print()
	
	vocab_a0 = ["exit, goodbye, bye, nevermind".split(", ")]
	vocab_a1 = ["what, whats, know, see, want, wants, show, tell, get, find, estimate, how, hows, where, wheres, can, please".split(", ")]
	vocab_a2 = [vocab_a1[0], "change, changes, changed, difference, differences".split(", ")]
	vocab_a3 = [vocab_a1[0], "trend, trends".split(", ")]
	vocab_a4 = [vocab_a1[0], "visual, visualize, visualise".split()]
	
	translate = jnp.array(translate)
	
	loop_active = True
	while loop_active:
		
		# What is the water level at (x,y,t)?
		# What is the change in water level between (x0, y0, t0) and (x1, y1, t1)?
		# What is the trend in water level between (x0, y0, t0) and (x1, y1, t1)?
		query = input(f"Q: ")
		query = query.lower()
		for c in ";'[]-=_+!@#$%^&/*~`<>?|:\"{}":
			query = query.replace(c,' ')
		words = query.split()
		coord_xyt = jnp.array(resolve_coord_xyt(query))
		
		if resolve_vocab_match(words, vocab_a0):
			print(f"A: Thank you for using the Champaign County Wells Expert System.")
			loop_active = False
		
		elif resolve_vocab_match(words, vocab_a1) and (len(coord_xyt) > 0):
			
			coord_z = translate[3,0] + h_fn(h_param, (coord_xyt - translate[:3,0]) / translate[:3,1]) * translate[3,1]
			for c,z in zip(coord_xyt, coord_z):
				print(f"A: At {c} the water level is {z:.2f}.")
			
			if (len(coord_xyt) > 1):
				
				if resolve_vocab_match(words, vocab_a2):
					for i in range(len(coord_xyt)):
						for j in range(i+1, len(coord_xyt)):
							print(f"A: The change (z1-z0) in water level between {coord_xyt[i]} and {coord_xyt[j]} is {coord_z[j]-coord_z[i]:.2f}.")
				
				if resolve_vocab_match(words, vocab_a3):
					for i in range(len(coord_xyt)):
						for j in range(i+1, len(coord_xyt)):
							
							trend_axis = jnp.linspace(0, 1, trend_res)
							trend_xyt = jnp.array([coord_xyt[i] + t * (coord_xyt[j] - coord_xyt[i]) for t in trend_axis])
							trend_z = translate[3,0] + h_fn(h_param, (trend_xyt - translate[:3,0]) / translate[:3,1]) * translate[3,1]
							
							print(f"A: Between {coord_xyt[i]} and {coord_xyt[j]}, the mean is {trend_z.mean():.4f} and the variance is {trend_z.var():.4f}. The trend is <figure>.")
							
							fig = plt.figure(figsize=(10, 6))
							gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 0], wspace=0.2, hspace=0.05)
							ax_main = plt.subplot(gs[0, 0])
							ax_main.plot(trend_axis, trend_z)
							ax_main.set_ylabel("Water level")
							ax_xtick_inc = int(trend_res*0.2)
							ax_main.set_xticks(trend_axis[::ax_xtick_inc], [f"({x:.1f},{y:.1f},{t:.1f})" for x,y,t in trend_xyt[::ax_xtick_inc]])
							ax_main.grid()
							ax_hist = plt.subplot(gs[0, 1], sharey=ax_main)
							ax_hist.hist(trend_z, bins=100, orientation='horizontal', color='grey', alpha=0.7)
							ax_hist.set_xticks([])
							plt.show()
				
				if resolve_vocab_match(words, vocab_a4):
					for i in range(len(coord_xyt)):
						for j in range(i+1, len(coord_xyt)):
							
							c0 = coord_xyt[i]
							c1 = coord_xyt[j]
							c_xmin, c_xmax = min([c0[0], c1[0]]), max([c0[0], c1[0]])
							c_ymin, c_ymax = min([c0[1], c1[1]]), max([c0[1], c1[1]])
							c_tmin, c_tmax = min([c0[2], c1[2]]), max([c0[2], c1[2]])
							axis_x = (jnp.linspace(c_xmin, c_xmax, 10) - translate[0,0]) / translate[0,1]
							axis_y = (jnp.linspace(c_ymin, c_ymax, 10) - translate[1,0]) / translate[1,1]
							axis_t = (jnp.linspace(c_tmin, c_tmax, 100) - translate[2,0]) / translate[2,1]
							sample_z =  translate[3,0] + sample_3d_model(h_fn, h_param, axis_t, axis_y, axis_x, batch_size=None) * translate[3,1]
							
							print("A: Here's a visual of that region.")
							
							animate_hydrology(
								sample_z,
								grid_extent=(axis_x.min(), axis_x.max(), axis_y.min(), axis_y.max()),
								cmap_contour='Blues_r',
								axis_ticks=True,
								origin=None,
								isolines=10,
								title_fn=lambda t: f"t={axis_t[t]:.2f}",
								clabel_fmt='%d'
							)
		
		else:
			print("A: Sorry, I don't understand. Can you rephrase your question?")
		
		print()
