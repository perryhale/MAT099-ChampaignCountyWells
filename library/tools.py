import re
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def resolve_coord_xyt(string):
	try:
		return [tuple(map(float, xyt)) for xyt in re.findall(r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+?)\s*\)', string)]
	except Exception:
		return []

def resolve_vocab_match(words, vocab):
	return all([any([w in v for w in words]) for v in vocab])


def expert_system(h_param, h_fn, trend_res=250):
	
	print("+======++======++======++======++======+")
	print("||Champaign County Wells Expert System||")
	print("+======++======++======++======++======+")
	print()
	print(f"Hello my name is WES Champaign, I'm an expert system intended to answer your questions about Champaign's water table. How can I help today?")
	#print("Try asking me a question. If you're unfamiliar with the actions possible within this portal, you can ask for help at any time.")
	print()
	
	vocab_a0 = ["exit, goodbye, bye, nevermind".split(", ")]
	vocab_a1 = ["what, whats, know, see, understand, want, wants, show, tell, get, find, estimate, how, hows, where, wheres".split(", ")]
	vocab_a2 = [vocab_a1[0], "change, changes, changed, difference, differences".split(", ")]
	vocab_a3 = [vocab_a1[0], "trend, trends".split(", ")]
	
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
			
			coord_z = h_fn(h_param, coord_xyt)
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
							trend_z = h_fn(h_param, trend_xyt)
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
		
		else:
			print("A: Sorry, I don't understand. Can you rephrase your question?")
		
		print()
