import time
import functools
from __future__ import print_function

def clock(func):
	@functools.wraps(func)
	def clocked(*args, **kwargs)
		t0 = time.time()
		result = func(*args, **kwargs)
		elapsed = time.time() - t0
		name = func.__name__
		args_list = list()
		if args:
			args_list.append(', '.join(repr(arg) for arg in args)
		if kwargs:
			pairs = ['%s=%r'%(k, w) for k, w in sorted(kwargs.items())]
			args_list.append(', '.join(pairs))
		args_string = ', '.join(args_list)
		print('[%0.8fs] %s(%s) -> %r '%(elapsed, name, args_str, result))
		return result
	return clocked