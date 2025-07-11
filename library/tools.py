import re
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def resolve_coords(string):
	return [tuple(map(float, xyt)) for xyt in re.findall(r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+?)\s*\)', string)]

def resolve_vocab_match(words, vocab):
	return all([any([w in v for w in words]) for v in vocab])


def expert_system(h_param, h_fn, trend_res=1000):
	
	print("+======++======++======++======++======++======+")
	print("||                                            ||")
	print("||    Champaign County Wells Expert System    ||")
	print("||                                            ||")
	print("+======++======++======++======++======++======+")
	print()
	print(f"Hello my name is WES Champaign, I'm an expert system intended to answer your questions about Champaign's water table. How can I help today?")
	#print("Try asking me a question. If you're unfamiliar with the actions possible within this portal, you can ask for help at any time.")
	print()
	
	vocab_a0 = ["exit, goodbye, bye, nevermind".split(", ")]
	vocab_a1 = ["what, whats, know, estimate, show, how, tell".split(", ")]
	vocab_a2 = [vocab_a1[0], "change, difference".split(", ")]
	vocab_a3 = [vocab_a1[0], "trend, ".split(", ")]
	
	loop_active = True
	while loop_active:
		
		# What is the water level at (x,y,t)?
		# What is the change in water level between (x0, y0, t0) and (x1, y1, t1)?
		# What is the trend in water level between (x0, y0, t0) and (x1, y1, t1)?
		query = input(f"Q: ")
		query = query.lower()
		for c in ";'[]-=_+!@#$%^&*~`<>?|:\"{}":
			query = query.replace(c,'')
		words = query.split()
		coords = jnp.array(resolve_coords(query))
		
		if resolve_vocab_match(words, vocab_a0):
			print(f"A: Thank you for using the Champaign County Wells Expert System.")
			loop_active = False
		
		elif resolve_vocab_match(words, vocab_a1) and (coords.shape[0]==1):
			z_pred = h_fn(h_param, coords)
			print(f"A: At {coords[0]} the water level is {z_pred[0]:.2f}.")
		
		elif resolve_vocab_match(words, vocab_a2) and (coords.shape[0]==2):
			z_pred = h_fn(h_param, coords)
			print(f"A: At {coords[0]} the water level is {z_pred[0]:.2f} and at {coords[1]} the water level is {z_pred[1]:.2f}. This means the change (z1-z0) in water level is {z_pred[1]-z_pred[0]:.2f}.")
		
		elif resolve_vocab_match(words, vocab_a3) and (coords.shape[0]==2):
			
			trend_axis = jnp.linspace(0, 1, trend_res)
			trend_coords = jnp.array([coords[0] + t * (coords[1] - coords[0]) for t in lin_axis])
			z_pred = h_fn(h_param, trend_coords)
			
			print(f"A: Between {coords[0]} and {coords[1]}, the mean is {z_pred.mean():.4f} and the variance is {z_pred.var():.4f}. The trend is <figure>.")
			
			fig = plt.figure(figsize=(10, 6))
			gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 0], wspace=0.2, hspace=0.05)
			ax_main = plt.subplot(gs[0, 0])
			ax_main.plot(trend_axis, z_pred)
			ax_main.set_ylabel("Water level")
			ax_xtick_inc = int(trend_res*0.2)
			ax_main.set_xticks(trend_axis[::ax_xtick_inc], [f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" for p in trend_coords[::ax_xtick_inc]])
			ax_main.grid()
			ax_hist = plt.subplot(gs[0, 1], sharey=ax_main)
			ax_hist.hist(z_pred, bins=100, orientation='horizontal', color='gray', alpha=0.7)
			ax_hist.set_xticks([])
			plt.show()
		
		else:
			print("A: Sorry, I don't understand. Can you rephrase your question?")
		
		print()
