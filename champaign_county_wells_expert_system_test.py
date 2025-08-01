import time
import pickle
import jax
import jax.numpy as jnp
from library.tools import *


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")


### functions

# type: () -> WellsExpertSystem
def get_test_expert_system():
	return WellsExpertSystem(None, lambda p,x: jnp.array([0.]*len(x)), testing=True)

# type: () -> bool
def test_resolve_coord_xyt():
	trials = [
		("(0,10,0)", [(0.0, 10.0, 0.0)]),
		("(0.,0.,0.)", [(0.0, 0.0, 0.0)]),
		("(.1,.1,1)", [(0.1, 0.1, 1.0)]),
		("(.1,  .1,.1)", [(0.1, 0.1, 0.1)]),
		("&abc(.1,.1,.1)&abc", [(0.1, 0.1, 0.1)]),
		("(1.2, 1, 0.1), (01.2,1,1)", [(1.2, 1.0, 0.1), (1.2, 1.0, 1.0)]),
		("(1.2, 1, 0.1) and (01.2,1,1)", [(1.2, 1.0, 0.1), (1.2, 1.0, 1.0)]),
		("(1.2, 1, 0.1)and(01.2,1,1)", [(1.2, 1.0, 0.1), (1.2, 1.0, 1.0)]),
		("abcd(a1,2,444)", []),
		("(1,2)", []),
		("(1,2 2, 4)", []),
		("(1,2,3", []),
		("1,2,3)", []),
		("(1,.2.,3)", []),
	]
	try:
		return all([resolve_coord_xyt(x)==y for x,y in trials])
	except Exception:
		return False


# type: () -> bool
def test_resolve_vocab_match():
	vocab = "test, north, epsilon".split(", ")
	trials = [
		("I'd like to test the vocabulary resolution function", True),
		("This string does not contain any known keywords", False),
		("This string contains the non-normalised keyword North", False),
		("This string contains multiple keywords: north and epsilon", True),
		("thekeywordsnorthtest are joined without spaces", True),
		("northerly is the same example", True)
	]
	try:
		return all([resolve_vocab_match(x, vocab)==y for x,y in trials])
	except Exception:
		return False


# type: () -> bool
def test_normalise_string():
	expert_system = get_test_expert_system()
	trials = [
		("What, where. how?", "what, where. how "),
		("you're", "you re"),
		("visualisation", "visualisation"),
		("*(0,0,0) AND (0,0,1)?", " (0,0,0) and (0,0,1) ")
	]
	try:
		return all([normalise_string(x, expert_system.vocab['punct'])==y for x,y in trials])
	except Exception:
		return False


# type: () -> bool
def test_wells_expert_system_respond():
	expert_system = get_test_expert_system()
	trials = [
		"What is the water level at (0,0,0)?",
		"What is the change in water level between (0,0,0) and (0,0,1)?",
		"What is the trend in water level over (0,0,0) and (0,0,1)?",
		"Can you show me a visualisation of (0.75,0.75,0) to (1,1,1)?",
		"This input does not contain any coordinates",
		"This input contains malformed coordinates (a1,1,.1.)"
	]
	try:
		for query in trials:
			expert_system.respond(query)
		return True
	except Exception as e:
		print(e)
		return False


### main

tests = [
	test_resolve_coord_xyt,
	test_resolve_vocab_match,
	test_normalise_string,
	test_wells_expert_system_respond
]
results = []

# run tests
for test in tests:
	t1 = time.time()
	r0 = test()
	print(f"Test {'passed' if r0 else 'failed'}: {test} [Run time: {(time.time()-t1)*1000:.2f}ms]")
	results.append(r0)

print(f"All passed: {all(results)}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
