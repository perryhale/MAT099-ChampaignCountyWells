import time
import re
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt; plt.style.use('classic')
import matplotlib.gridspec as gridspec
from library.models.nn import sample_3d_model
from library.visual import animate_hydrology


PATTERN_COORD_XYT = re.compile(r'\(\s*(-?(?:\d+\.\d*|\.\d+|\d+))\s*,\s*(-?(?:\d+\.\d*|\.\d+|\d+))\s*,\s*(-?(?:\d+\.\d*|\.\d+|\d+))\s*\)') # match for signed 3-tuples (a,b,c) including leading/trailing full-stop


def resolve_coord_xyt(string):
	return [tuple(map(float, xyt)) for xyt in PATTERN_COORD_XYT.findall(string)]

def resolve_vocab_match(string, vocab):
	return any([w in string for w in vocab])

def normalise_string(string, punct):
	string = string.lower()
	for c in punct:
		string = string.replace(c,' ')
	return string


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

def print_word_overflow(string, width=64, delay=1/120, newline=""):
	
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


class WellsExpertSystem():
	
	# type: (WellsExpertSystem, pytree, (pytree, jnp.array)->jnp.array, dict[list[str]], list[tuple[float*2]], int, tuple[int*3], str, int, float, bool) -> None
	def __init__(self, h_param, h_fn, vocab=None, transform=[(0,1)]*4, res_trend=250, res_vis=(10,10,100), unit="", width=64, delay=1/120, testing=False):
		self.h_param = h_param
		self.h_fn = h_fn
		self.transform = jnp.array(transform)
		self.res_trend = res_trend
		self.res_vis = res_vis
		self.unit = unit
		self.width = width
		self.delay = delay
		if vocab is None:
			self.vocab = {
				'punct': list(";'[]=_+!@#$%^&/*~`<>?|:\"{}"),
				'exit': "exit, bye, nevermind".split(", "),
				'change': "change, difference".split(", "),
				'trend': "trend, line".split(", "),
				'visual': "visual, volume".split(", ")
			}
		else:
			required_keys = "punct, exit, change, trend, visual".split(", ")
			assert all([key in list(vocab.keys()) for key in required_keys]), f"Provided vocab dictionary must specify {required_keys}."
			assert all([type(x)==list and type(x[0])==str for x in vocab.values()]), "Provided vocab dictionary values must be type list<str>."
			self.vocab = vocab
		
		self.testing = testing
	
	# type: (WellsExpertSystem) -> None
	def introduce(self):
		if not self.testing:
			print_in_box("Champaign County\nWells Expert System", segment_width=8, max_width=self.width-8)
			print()
			print_word_overflow(" ".join([
				"Hello my name is WES Champaign, I'm an expert system intended to answer your questions about Champaign's water table.",
				"I use a 3D coordinate system for positions on the ground in time (x,y,t).",
				"You can ask me about the height of the water level at any coordinates and about both the change and trend in-between the water levels at any coordinates.",
				"You can also ask me to visualise the 3D volume bound by any coordinates.",
				"If you're asking about lines or volumes (that's trends and visuals) then I'll also report the mean and variance.",
				"If you'd like to leave at any time, just say bye or exit.",
				"How can I help today?"
			]), width=self.width, delay=self.delay)
			print()
	
	# type: (WellsExpertSystem, str) -> None
	def respond(self, query):
		
		coord_xyt = jnp.array(resolve_coord_xyt(query))
		if (len(coord_xyt) > 0):
			
			coord_z = self.transform[3,0] + self.h_fn(self.h_param, (coord_xyt - self.transform[:3,0]) / self.transform[:3,1]) * self.transform[3,1]
			for xyt,z in zip(coord_xyt, coord_z):
				if not self.testing: print_word_overflow(
					f"A: At {vec_to_string(xyt)} the water level is {z:.2f}{self.unit}.",
					width=self.width, delay=self.delay, newline=" "*3
				)
			
			if (len(coord_xyt) > 1):
				
				if resolve_vocab_match(query, self.vocab['change']):
					for i in range(len(coord_xyt)):
						for j in range(i+1, len(coord_xyt)):
							c0 = coord_xyt[i]
							c1 = coord_xyt[j]
							if not self.testing: print_word_overflow(
								f"A: The change (z1-z0) in water level between {vec_to_string(c0)} and {vec_to_string(c1)} is {coord_z[j]-coord_z[i]:.2f}{self.unit}.",
								width=self.width, delay=self.delay, newline=" "*3
							)
				
				if resolve_vocab_match(query, self.vocab['trend']):
					for i in range(len(coord_xyt)):
						for j in range(i+1, len(coord_xyt)):
							
							c0 = coord_xyt[i]
							c1 = coord_xyt[j]
							trend_axis = jnp.linspace(0, 1, self.res_trend)
							trend_xyt = jnp.array([c0 + t * (c1 - c0) for t in trend_axis])
							trend_z = self.transform[3,0] + self.h_fn(self.h_param, (trend_xyt - self.transform[:3,0]) / self.transform[:3,1]) * self.transform[3,1]
							
							if not self.testing: print_word_overflow(" ".join([
								f"A: Here's the trend on the line between {vec_to_string(c0)} and {vec_to_string(c1)}: <figure>.",
								f"On this line the mean is {trend_z.mean():.2f}{self.unit} and the variance is {trend_z.var():.2f}{self.unit}."
							]), width=self.width, delay=self.delay, newline=" "*3)
							
							fig = plt.figure(figsize=(7, 4))
							gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 0], wspace=0.2, hspace=0.05)
							ax_main = plt.subplot(gs[0, 0])
							ax_main.plot(trend_axis, trend_z)
							ax_main.set_ylabel("Water level" + f"({self.unit})" if self.unit!="" else "")
							ax_xtick_inc = int(self.res_trend/3)
							ax_main.set_xticks(trend_axis[::ax_xtick_inc], [vec_to_string(v, d=1) for v in trend_xyt[::ax_xtick_inc]])
							ax_main.grid()
							ax_hist = plt.subplot(gs[0, 1], sharey=ax_main)
							ax_hist.hist(trend_z, bins=100, orientation='horizontal', color='grey', alpha=0.7)
							ax_hist.set_xticks([])
							if not self.testing: plt.show()
							else: plt.clf()
				
				if resolve_vocab_match(query, self.vocab['visual']):
					for i in range(len(coord_xyt)):
						for j in range(i+1, len(coord_xyt)):
							
							c0 = coord_xyt[i]
							c1 = coord_xyt[j]
							c_xmin, c_xmax = min([c0[0], c1[0]]), max([c0[0], c1[0]])
							c_ymin, c_ymax = min([c0[1], c1[1]]), max([c0[1], c1[1]])
							c_tmin, c_tmax = min([c0[2], c1[2]]), max([c0[2], c1[2]])
							axis_x = (jnp.linspace(c_xmin, c_xmax, self.res_vis[0]) - self.transform[0,0]) / self.transform[0,1]
							axis_y = (jnp.linspace(c_ymin, c_ymax, self.res_vis[1]) - self.transform[1,0]) / self.transform[1,1]
							axis_t = (jnp.linspace(c_tmin, c_tmax, self.res_vis[2]) - self.transform[2,0]) / self.transform[2,1]
							sample_z =  self.transform[3,0] + sample_3d_model(self.h_fn, self.h_param, axis_t, axis_y, axis_x, batch_size=None) * self.transform[3,1]
							
							if not self.testing: print_word_overflow(" ".join([
								f"A: Here's a visualisation of the volume bound by {vec_to_string(c0)} and {vec_to_string(c1)}: <figure>.",
								f"Within this volume the mean is {sample_z.mean():.2f}{self.unit} and the variance is {sample_z.var():.2f}{self.unit}."
							]), width=self.width, delay=self.delay, newline=" "*3)
							
							if not self.testing: animate_hydrology(
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
			if not self.testing: print_word_overflow(" ".join([
				"A: Sorry, I don't understand.",
				"Can you rephrase your question?",
				"Make sure to provide 3D coordinates (x,y,t) in decimal format.",
			]), width=self.width, delay=self.delay, newline=" "*3)
		
		if not self.testing: print()
	
	# type: (WellsExpertSystem) -> None
	def control_loop(self):
		self.introduce()
		while True:
			query = input(f"Q: ")
			query = normalise_string(query, self.vocab['punct'])
			if resolve_vocab_match(query, self.vocab['exit']):
				if not self.testing: print_word_overflow(
					f"A: Thank you for using the Champaign County Wells Expert System.",
					width=self.width, delay=self.delay, newline=" "*3
				)
				print()
				break
			else:
				self.respond(query)
