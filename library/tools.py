import time
import re
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt; plt.style.use('classic')
import matplotlib.gridspec as gridspec
from library.models.nn import sample_3d_model
from library.visual import animate_hydrology


PATTERN_COORD_XYT = re.compile(r'\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)') # match for signed 3-tuples (a,b,c)


def resolve_coord_xyt(string):
	return [tuple(map(float, xyt)) for xyt in PATTERN_COORD_XYT.findall(string)]

def resolve_vocab_match(string, vocab):
	return any([w in string for w in vocab])


def print_in_box(text, segment_width=8, max_width=56):
	
	segment = f"+{'=' * (segment_width-2)}+"
	top_border = segment * (max_width // len(segment))
	print(top_border)
	
	for line in text.splitlines():
		trimmed = line.strip()
		padding = len(top_border) - len(trimmed) - 4
		left_pad = padding // 2
		right_pad = padding - left_pad
		print(f"||{' ' * left_pad}{trimmed}{' ' * right_pad}||")
	
	print(top_border)

def print_word_overflow(string, width=64, delay=1/120, newline=" "*3):
	
	words = string.split(' ')
	column = 0
	
	for i in range(len(words)-1):
		print(words[i], end=" ", flush=True)
		column += len(words[i])+1
		if (column+len(words[i+1]) >= width):
			print()
			print(newline, end="")
			column = len(newline)
		if delay > 0:
			time.sleep(delay)
	
	print(words[-1])

def vec_to_string(arr, sep=",", d=2):
	value_str = sep.join([f'{v:.{d}f}' for v in arr])
	return f"({value_str})"


def expert_system(h_param, h_fn, vocab=None, translate=[(0,1)]*4, res_trend=250, res_vis=(10,10,100), unit=""):
	
	if vocab is None:
		vocab = {
			'punct': ";'[]=_+!@#$%^&/*~`<>?|:\"{}",
			'exit': "exit, bye, nevermind".split(", "),
			'change': "change, difference".split(", "),
			'trend': "trend, line".split(", "),
			'visual': "visual, volume".split(", ")
		}
	else:
		required_keys = "punct, exit, change, trend, visual".split(", ")
		assert required_keys in list(vocab.keys()), f"Provided vocab dictionary must specify {required_keys}."
	
	translate = jnp.array(translate)
	
	print_in_box("Champaign County\nWells Expert System")
	print()
	print_word_overflow(" ".join([
		"Hello my name is WES Champaign, I'm an expert system intended to answer your questions about Champaign's water table.",
		"I use a 3D coordinate system for positions on the ground in time (x,y,t).",
		"You can ask me about the height of the water level at any coordinates and about both the change and trend in-between the water levels at any coordinates.",
		"You can also ask me to visualise the 3D volume bound by any coordinates.",
		"If you're asking about lines or volumes (that's trends and visuals) then I'll also report the mean and variance.",
		"If you'd like to leave at any time, just say bye or exit.",
		"How can I help today?"
	]), newline="")
	print()
	
	while True:
		
		# What is the water level at (x,y,t)?
		# What is the change in water level between (x0, y0, t0) and (x1, y1, t1)?
		# What is the trend in water level between (x0, y0, t0) and (x1, y1, t1)?
		# Can you show me a visualisation of (x0, y0, t0) to (x1, y1, t1)?
		query = input(f"Q: ")
		query = query.lower()
		for c in vocab['punct']:
			query = query.replace(c,' ')
		
		if resolve_vocab_match(query, vocab['exit']):
			print_word_overflow(f"A: Thank you for using the Champaign County Wells Expert System.")
			break
		
		else:
			coord_xyt = jnp.array(resolve_coord_xyt(query))
			if (len(coord_xyt) > 0):
				
				coord_z = translate[3,0] + h_fn(h_param, (coord_xyt - translate[:3,0]) / translate[:3,1]) * translate[3,1]
				for xyt,z in zip(coord_xyt, coord_z):
					print_word_overflow(f"A: At {vec_to_string(xyt)} the water level is {z:.2f}{unit}.")
				
				if (len(coord_xyt) > 1):
					
					if resolve_vocab_match(query, vocab['change']):
						for i in range(len(coord_xyt)):
							for j in range(i+1, len(coord_xyt)):
								c0 = coord_xyt[i]
								c1 = coord_xyt[j]
								print_word_overflow(f"A: The change (z1-z0) in water level between {vec_to_string(c0)} and {vec_to_string(c1)} is {coord_z[j]-coord_z[i]:.2f}{unit}.")
					
					if resolve_vocab_match(query, vocab['trend']):
						for i in range(len(coord_xyt)):
							for j in range(i+1, len(coord_xyt)):
								
								c0 = coord_xyt[i]
								c1 = coord_xyt[j]
								trend_axis = jnp.linspace(0, 1, res_trend)
								trend_xyt = jnp.array([c0 + t * (c1 - c0) for t in trend_axis])
								trend_z = translate[3,0] + h_fn(h_param, (trend_xyt - translate[:3,0]) / translate[:3,1]) * translate[3,1]
								
								print_word_overflow(" ".join([
									f"A: Here's the trend on the line between {vec_to_string(c0)} and {vec_to_string(c1)}: <figure>.",
									f"On this line the mean is {trend_z.mean():.2f}{unit} and the variance is {trend_z.var():.2f}{unit}."
								]))
								
								fig = plt.figure(figsize=(7, 4))
								gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 0], wspace=0.2, hspace=0.05)
								ax_main = plt.subplot(gs[0, 0])
								ax_main.plot(trend_axis, trend_z)
								ax_main.set_ylabel("Water level" + f"({unit})" if unit!="" else "")
								ax_xtick_inc = int(res_trend/3)
								ax_main.set_xticks(trend_axis[::ax_xtick_inc], [vec_to_string(v, d=1) for v in trend_xyt[::ax_xtick_inc]])
								ax_main.grid()
								ax_hist = plt.subplot(gs[0, 1], sharey=ax_main)
								ax_hist.hist(trend_z, bins=100, orientation='horizontal', color='grey', alpha=0.7)
								ax_hist.set_xticks([])
								plt.show()
					
					if resolve_vocab_match(query, vocab['visual']):
						for i in range(len(coord_xyt)):
							for j in range(i+1, len(coord_xyt)):
								
								c0 = coord_xyt[i]
								c1 = coord_xyt[j]
								c_xmin, c_xmax = min([c0[0], c1[0]]), max([c0[0], c1[0]])
								c_ymin, c_ymax = min([c0[1], c1[1]]), max([c0[1], c1[1]])
								c_tmin, c_tmax = min([c0[2], c1[2]]), max([c0[2], c1[2]])
								axis_x = (jnp.linspace(c_xmin, c_xmax, res_vis[0]) - translate[0,0]) / translate[0,1]
								axis_y = (jnp.linspace(c_ymin, c_ymax, res_vis[1]) - translate[1,0]) / translate[1,1]
								axis_t = (jnp.linspace(c_tmin, c_tmax, res_vis[2]) - translate[2,0]) / translate[2,1]
								sample_z =  translate[3,0] + sample_3d_model(h_fn, h_param, axis_t, axis_y, axis_x, batch_size=None) * translate[3,1]
								
								print_word_overflow(" ".join([
									f"A: Here's a visualisation of the volume bound by {vec_to_string(c0)} and {vec_to_string(c1)}: <figure>.",
									f"Within this volume the mean is {sample_z.mean():.2f}{unit} and the variance is {sample_z.var():.2f}{unit}."
								]))
								
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
				print_word_overflow(" ".join([
					"A: Sorry, I don't understand.",
					"Can you rephrase your question?",
					"Make sure to provide 3D coordinates (x,y,t) in decimal format.",
				]))
		
		print()
