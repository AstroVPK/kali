import sys, os, argparse, time, pdb, math, thread, cPickle
import numpy as np, numba
from scipy.spatial.distance import euclidean
import libcarma
import carmcmc
from matplotlib.pyplot import subplots, show, figure
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from multiprocessing import Process

#Defaults:
dt = 0.1
T = 10
numCadences = int(T/dt)
IR = False
tolIR = 1.0e-3
fracIntrinsicVar = 0.1
fracNoiseToSignal = 1.0e-8
maxSigma = 2.0
minTimescale = 5.0e-1
maxTimescale = 2.0
ar_roots = [-0.033642081+0.0254j, -0.033642081-0.0254j]
ar_roots = [-0.104575, -0.033642081+0.0254j, -0.033642081-0.0254j]
#ar_roots = [-0.033642081, -0.033642081]
#ar_roots = [-0.033642081+0.254j, -0.033642081-0.254j, -0.052, -0.0042]
#ar_roots = [-0.033642081]
#sigma = 7.0e-9
sigma = 2.0e-9
#ma_roots = [-5.833333]
ma_roots = [-3.833333, -0.001853]
#ma_roots = []
#ma_roots = [-4.833333, -2.46, 0.000245]
NWALKERS = 20#100
NSTEPS = 200*2
CLOCK_FLAG = 1

class clock:

	def __init__(self):
		self.CLOCK_FLAG = 1
	
	def start(self):
		self.run()

	def stop(self):
		self.CLOCK_FLAG = 0

	def run(self):
		T0 = time.time()
		#anim_chars = '|/-\\|/-'
		#anim_chars = '>v<^'
		anim = '*+._.+*"'
		anim_chars = [2*"".join((anim[i:],anim[:i])) for i in xrange(len(anim))]
		anim_N = len(anim_chars)
		anim_n = 0
		cycle_n = 0
		s = ''
		m = ''
		h = ''
		T = 0
		string = ''
		print "Clock Started"
		sys.__stdout__.flush()
		while self.CLOCK_FLAG:
			time.sleep(0.05)
			if (time.time() - T0 - T) > 0.1:
				T = time.time() - T0
				s = 'o'.join((str(int(T) % 60).zfill(2), str(T-int(T))[2:3][::-1].zfill(1)[::-1]))
				m = str((int(T) / 60) % 60).zfill(2)
				h = str(int(T) / 3600).zfill(2)
				string = ':'.join((h,m,s))
				if cycle_n == 2:
					c_anim = anim_chars[anim_n].join('[]')
					string = ' '.join((string, c_anim))
					anim_n += 1
					cycle_n = 0
				cycle_n += 1
				string = ''.join((string,'\r'))
				sys.__stdout__.write(string)
				sys.__stdout__.flush()
			
				anim_n %= anim_N

def run_clock():
	c = clock()
	c.start()

def start_clock():
	C = Process(target = run_clock)
	C.daemon = True
	C.start()
	return C


def plot_params(p1, p2, loglik, title = None, ax = None):

	if ax is None:
		fig, ax = subplots(1,1)
	if title is not None:
		ax.set_title(title)
	c = np.exp(loglik - max(loglik))
	ax.scatter(p1, p2, c = c, alpha = 0.1, marker = 'o', edgecolor = 'none')
	ax.set_xlim(min(p1), max(p1))
	ax.set_ylim(min(p2), max(p2))

	return ax

def getErrors(theta_exp, theta_0):
	'''computer the euclidian distance between the experimental params and the original params'''
	return euclidean(theta_exp/theta_0, np.ones(len(theta_0)))
	

def getTheta(ar_roots, ma_roots, sigma):
	'''Return theta for Libcarma'''
	ma_poly = sigma*np.poly(ma_roots) #TODO Change * to /
	ar_poly = np.poly(ar_roots)
	theta = np.concatenate((ar_poly[1:], ma_poly))
	p = len(ar_roots)
	q = len(ma_roots)
		
	return theta, p, q

def downSample(N, n):
	'''return of mask of n random points out of N total'''
	return np.random.choice(np.arange(N), n)

def generateLC(ar_roots, ma_roots, sigma, module = 'libcarma'):
	'''generate a mock LC from either libcarma or carmcmc'''
	if module == 'libcarma':
		theta, p, q = getTheta(ar_roots, ma_roots, sigma)	
		
		newTask = libcarma.basicTask(p, q, nwalkers = NWALKERS, nsteps = NSTEPS)
		newTask.set(dt, theta)

		newLC = newTask.simulate(numCadences)
		newTask.observe(newLC)
		#print "LnLike:", newTask.logPosterior(newLC)
		#t, y, yerr = newLC.t, newLC.y, newLC.yerr
	
	else:	

		t = np.arange(0, T, dt)
		ma_coefs = np.poly(ma_roots) #TODO Ditto
		y = carmcmc.carma_process(t, sigma**2, np.array(ar_roots), ma_coefs = ma_coefs)
		yerr = np.absolute(np.random.randn(len(y))*fracNoiseToSignal*(y - np.mean(y)))
		
	#return t, y, yerr
	return newLC

def fitCARMA(LC, p, q, module = 'libcarma'):
	'''fit a LC to a CARMA model and return the sample or task object'''
	
	if module == 'libcarma':
		#guess = np.absolute(np.random.randn(p+q+1)*np.mean(LC.y)*0.01)
		guess, p, q = getTheta(ar_roots, ma_roots, sigma)
		#LC = libcarma.basic#LC(len(t), tolIR = tolIR, fracIntrinsicVar = fracIntrinsicVar, fracNoiseToSignal = fracNoiseToSignal, maxSigma = maxSigma, minTimescale = minTimescale, maxTimescale = maxTimescale)
		#LC.y, #LC.yerr, #LC.t = y, yerr, t
		#LC._dt = min(#LC.t[1:] - #LC.t[:-1])
		#new#LC = #LC.copy()
		#LC = new#LC
		newTask = libcarma.basicTask(p, q, nwalkers = NWALKERS, nsteps = NSTEPS)
		LC.maxTimescale = 2.0
		newTask.fit(LC, guess)
		return newTask
	
	else:
		t, y, yerr = LC.t, LC.y, LC.yerr
		model = carmcmc.CarmaModel(t, y, yerr, p = p, q = q)
		sample = model.run_mcmc(NWALKERS*NSTEPS/2)
		return sample		

def walkerPlot(chain, index, loglik, title = None):

	data = chain[index,:,:]
	m, n = data.shape
	step_index = np.arange(n)
	lik = np.exp(loglik - np.max(loglik.flatten()))

	fig, ax = subplots(1,1)
	if title is not None:
		if type(title) is str:
			fig.suptitle(title)
	for i in xrange(m):
		ax.scatter(step_index, data[i,:], c = lik[i,:], marker = 'o', edgecolor = 'none', alpha = 0.1)
	
	ax.plot(step_index, np.median(data, axis = 0), color = 'r', lw = 2)

	ax.set_xlim(0,n)

	return ax


def errorPlot(chain, loglik, theta, title = None):

	l,m,n = chain.shape
	data = np.array([[getErrors(chain[:,i,j], theta) for i in xrange(m)] for j in xrange(n)]).T
	step_index = np.arange(n)
	lik = np.exp(loglik - np.max(loglik.flatten()))
	fig, ax = subplots(1,1)
	if title is not None:
		if type(title) is str:
			fig.suptitle(title)
	for i in xrange(m):
		ax.scatter(step_index, data[i,:], c = lik[i,:], marker = 'o', edgecolor = 'none', alpha = 0.1)
	ax.plot(step_index, np.median(data, axis = 0), color = 'r', lw = 2)

	return ax

def generateTheta(obj):
	'''generate the theta_exp values for an obj'''
	if type(obj) is not carmcmc.CarmaSample:
		chain = obj.Chain
		theta_exp = np.array([chain[i,:,obj.nsteps/2:].flatten() for i in xrange(chain.shape[0])]).T
		lnlk = np.array(obj.LnPosterior[:,obj.nsteps/2:]).flatten()
	else:
		ar_coefs = obj.get_samples('ar_coefs')[:,1:]
		ma_coefs = obj.get_samples('ma_coefs')
		sigma = obj.get_samples('sigma')
		poly_coefs = ma_coefs*sigma
		theta_exp = np.concatenate((ar_coefs, poly_coefs), axis = 1)
		lnlk = obj.get_samples('logpost')[:,0]

	return theta_exp, lnlk

def compareLnLik(obj1, obj2, LC):
	t1, l1 = generateTheta(obj1)
	t2, l2 = generateTheta(obj2)
	n,m = t2.shape
	print l1.shape, l2.shape
	for i in xrange(n):
		obj1.set(dt, t2[i])
		l1[i] = obj1.logPosterior(LC)
	fig, ax = subplots(1,1)
	nbins = int(np.sqrt(len(l2)))
	#ax.hist([l1, l2], bins = nbins, histtype = 'stepfilled', stacked =True, normed = True, color = ['#A0A0DC','#DCA0A0'])
	ax.hist(l1, bins = nbins, histtype = 'stepfilled', stacked =True, normed = True, color = '#A0A0DC',alpha = 0.5)
	ax.hist(l2, bins = nbins, histtype = 'stepfilled', stacked =True, normed = True, color = '#DCA0A0', alpha = 0.5)
	return ax

def getBins(data):
	q75, q25 = np.percentile(data, [75, 25]) #start by finding number of bins
	IQR = q75 - q25
	nbins = int((max(data) - min(data))/(2*IQR*len(data)**(-1.0/3)))
	return nbins

def trianglePlot(obj):

	theta, lnlk = generateTheta(obj)
	lnlk = np.exp(lnlk - max(lnlk))
	m, n = theta.shape
	fig = figure(figsize = (n*5, n*5), dpi = 100)
	for i in xrange(n):
		for j in xrange(i+1):
			if i == j:
				nbins = getBins(theta[:,i])
				ax = fig.add_subplot(n, n, i*n+j+1)
				ax.set_title(r'$\theta_{%i}$' % (j + 1))
				num, bins, patches = ax.hist(theta[:,i], bins = nbins, histtype = 'stepfilled', normed = True, color = '#C0C0DC')
				ax.autoscale_view(False, False, False)
				ax.autoscale(False, False, False)
				print j+1, min(bins), max(bins)
				ax.set_xlim(min(bins), max(bins))
			else:
				yAxis = None if j == 0 else fig.axes[i*(i+1)/2]
				xAxis = fig.axes[(j+1)*(j+2)/2-1]
				ax = fig.add_subplot(n, n, n*i+j+1, sharex = xAxis, sharey = yAxis)
				ax.scatter(theta[:,j], theta[:,i], c = lnlk, marker = 'o', edgecolor = 'none', alpha = 0.1)
				#ax.set_xlim(min(theta[:,j]), max(theta[:,j]))
				ax.set_ylim(min(theta[:,i]), max(theta[:,i]))
			if i != n - 1:
				map(lambda x: x.set_visible(False), ax.get_xticklabels())
			elif j < n:
				ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.xaxis.set_major_locator(MaxNLocator(nbins = 6, prune = 'lower'))
			if j != 0:
				map(lambda y: y.set_visible(False), ax.get_yticklabels())
			elif i > 0:
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.yaxis.set_major_locator(MaxNLocator(nbins = 6, prune = 'upper'))
	fig.subplots_adjust(hspace = 0)
	fig.subplots_adjust(wspace = 0)
	return ax


def trianglePlotCompare(obj1, obj2, theta):

	theta1, lnlk1 = generateTheta(obj1)
	theta2, lnlk2 = generateTheta(obj2)

	theta1 = 1.0/np.array([libcarma.roots(3, 2, t1) for t1 in theta1]).real
	theta2 = 1.0/np.array([libcarma.roots(3, 2, t2) for t2 in theta2]).real
	theta = 1.0/np.array(libcarma.roots(3, 2, theta)).real

	lnlk1 = np.exp(lnlk1 - max(lnlk1))
	lnlk2 = np.exp(lnlk2 - max(lnlk2))
	pos1 = lnlk1.argmax()
	pos2 = lnlk2.argmax()

	m, n = theta1.shape
	fig = figure(figsize = (n*5, n*5), dpi = 100)
	for i in xrange(n):
		for j in xrange(i+1):
			if i == j:
				nbins = getBins(np.concatenate((theta1[:,i], theta2[:,i]), axis = 0))
				ax = fig.add_subplot(n, n, i*n+j+1)
				ax.set_title(r'$\theta_{%i}$' % (j + 1))
				num, bins, patches = ax.hist([theta1[:,i], theta2[:,i]], bins = nbins, histtype = 'stepfilled', normed = True, color = ['#A0A0DC','#DCA0A0'], alpha = 1.0, stacked = True)
				ax.axvline(theta[i], color = 'g')
				ax.axvline(theta1[pos1,i], color = 'b')
				ax.axvline(theta2[pos2,i], color = 'r')
				#nbins = getBins(theta2[:,i])
				#num, bins2, patches = ax.hist(theta2[:,i], bins = nbins, histtype = 'stepfilled', normed = True, color = '#DCA0A0', alpha = 0.5)
				ax.autoscale_view(False, False, False)
				ax.autoscale(False, False, False)
				#ax.set_xlim(min([min(bins1), min(bins2)]), max([max(bins1), max(bins2)]))
				ax.set_xlim(min(bins), max(bins))
			else:
				yAxis = None if j == 0 else fig.axes[i*(i+1)/2]
				xAxis = fig.axes[(j+1)*(j+2)/2-1]
				ax = fig.add_subplot(n, n, n*i+j+1, sharex = xAxis, sharey = yAxis)
				ax.scatter(theta1[:,j], theta1[:,i], c = lnlk1, marker = 'o', edgecolor = 'none', alpha = 0.5, cmap = 'cool')
				ax.scatter(theta2[:,j], theta2[:,i], c = lnlk2, marker = 'o', edgecolor = 'none', alpha = 0.5, cmap = 'autumn')
				ax.axvline(theta[j], color = 'g')
				ax.axhline(theta[i], color = 'g')
				ax.axhline(theta1[pos1,i], color = 'b')
				ax.axvline(theta1[pos1,j], color = 'b')
				ax.axhline(theta2[pos2,i], color = 'r')
				ax.axvline(theta2[pos2,j], color = 'r')
				#ax.set_xlim(min(theta[:,j]), max(theta[:,j]))
				ax.set_ylim(min([min(theta1[:,i]), min(theta2[:,i])]), max([max(theta1[:,i]), max(theta2[:,i])]))
			if i != n - 1:
				map(lambda x: x.set_visible(False), ax.get_xticklabels())
			elif j < n:
				ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.xaxis.set_major_locator(MaxNLocator(nbins = 6, prune = 'lower'))
			if j != 0:
				map(lambda y: y.set_visible(False), ax.get_yticklabels())
			elif i > 0:
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.yaxis.set_major_locator(MaxNLocator(nbins = 6, prune = 'upper'))
	fig.subplots_adjust(hspace = 0)
	fig.subplots_adjust(wspace = 0)
	return ax
	

	
def Analyze(obj, theta):
	'''quick and dirty analysis of retruned params'''
	theta = 1.0/np.array(libcarma.roots(3,2,theta).real)
	if type(obj) is not carmcmc.CarmaSample:
		chain = obj.Chain
		theta_exp = np.array([chain[i,:,obj.nsteps/2:].flatten() for i in xrange(chain.shape[0])]).T
		lnlk = np.array(obj.LnPosterior[:,obj.nsteps/2:]).flatten()
		title = 'libcarma'
		print "Libcarma Errors:"
	else:
		ar_coefs = obj.get_samples('ar_coefs')[:,1:]
		ma_coefs = obj.get_samples('ma_coefs')
		sigma = obj.get_samples('sigma')
		poly_coefs = ma_coefs*sigma
		theta_exp = np.concatenate((ar_coefs, poly_coefs), axis = 1)
		lnlk = obj.get_samples('loglik')[:,0]
		title = 'carmcmc'
		print "Carmcmc Errors:"
	theta_exp = 1.0/np.array([libcarma.roots(3,2,t) for t in theta_exp]).real
	pos = lnlk.argmax()
	print "    Map:", getErrors(theta_exp[pos], theta)
	print "    Med:", getErrors(np.median(theta_exp, axis = 0), theta)

	ax1 = plot_params(theta_exp[:,0], theta_exp[:,1], lnlk, title = '\n'.join((title, "ar_coefs")))
	ax2 = plot_params(theta_exp[:,3], theta_exp[:,4], lnlk, title = '\n'.join((title, "ma_coefs"))) 

	ax1.axvline(theta_exp[pos,0], color = 'g') #Map values
	ax1.axhline(theta_exp[pos,1], color = 'g')

	ax1.axvline(np.median(theta_exp[:,0], axis = 0), color = 'b') #median values
	ax1.axhline(np.median(theta_exp[:,1], axis = 0), color = 'b')

	ax1.axvline(theta[0], color = 'r') #True values
	ax1.axhline(theta[1], color = 'r')

	

	ax2.axvline(theta_exp[pos,3], color = 'g') # Map values
	ax2.axhline(theta_exp[pos,4], color = 'g')

	ax2.axvline(np.median(theta_exp[:,3], axis = 0), color = 'b') #median values
	ax2.axhline(np.median(theta_exp[:,4], axis = 0), color = 'b')

	ax2.axvline(theta[3], color = 'r') #True values
	ax2.axhline(theta[4], color = 'r')

	return theta_exp[pos], np.median(theta_exp, axis = 0)


def carmcmc_test(LC):

	fig, ax = subplots(1,1)
	lnlk = sample.get_samples('loglik')[:,0]
	print "Lnlike:", max(lnlk)
	ax.scatter(sample.get_samples('ar_coefs')[:,1], sample.get_samples('ar_coefs')[:,2], c = max(lnlk) - lnlk)
	pos = sample.get_samples('logpost').argmax()
	ax.axhline(sample.get_samples('ar_coefs')[pos,2])
	ax.axvline(sample.get_samples('ar_coefs')[pos,1])

	ax.axhline(np.median(sample.get_samples('ar_coefs')[:,2]), color = 'k')
	ax.axvline(np.median(sample.get_samples('ar_coefs')[:,1]), color = 'k')
	ar_poly = np.poly(ar_roots)
	ax.axhline(ar_poly[2], color = 'r')
	ax.axvline(ar_poly[1], color = 'r')

	print "Carmcmc"
	print "Errors"
	print "Map:"
	print "    a1:",abs(sample.get_samples('ar_coefs')[pos,1] - ar_poly[1])/(ar_poly[1])
	print "    a2:",abs(sample.get_samples('ar_coefs')[pos,2] - ar_poly[2])/(ar_poly[2])
	print "Med:"
	print "    a1:",abs(np.median(sample.get_samples('ar_coefs')[:,1]) - ar_poly[1])/(ar_poly[1])
	print "    a2:",abs(np.median(sample.get_samples('ar_coefs')[:,2]) - ar_poly[2])/(ar_poly[2])



	show(False)
	

	return t, y	
	
def libcarma_test():

	chain = newTask.Chain#.reshape(-1, newTask._ndims, order = 'C')
	ar_coef_1 = chain[0,:,newTask.nsteps/2:].flatten()
	ar_coef_2 = chain[1,:,newTask.nsteps/2:].flatten()
	#ar_coef_1 = np.array(list(ar_coef_1))
	#ar_coef_2 = np.array(list(ar_coef_2))

	lnlk = newTask.LnPosterior
	theta_exp = np.array([chain[i,:,newTask.nsteps/2:].flatten() for i in xrange(chain.shape[0])]).T
	print theta_exp.shape
	fig, ax = subplots(5,1)
	steps = np.arange(len(chain[0,0,:]))
	print lnlk.shape
	print chain[0,:,:].shape

	for j in xrange(chain.shape[0]):
		for i in xrange(NWALKERS):
			ax[j+1].scatter(steps, chain[j,i,:], c = np.exp(lnlk[i,:] - max(lnlk[i,:])), alpha = 0.1, edgecolor = 'none', marker = 'o')
		ax[j+1].plot(np.median(chain[j,:,:], axis = 0), color = 'r', lw = 2)
	lnlk = np.array(newTask.LnPosterior[:,newTask.nsteps/2:])
	lnlk = lnlk.flatten()

	ax[1].axhline(ar_poly[1])
	ax[2].axhline(ar_poly[2])
	ax[0].scatter(ar_coef_1, ar_coef_2, c = np.exp(lnlk - max(lnlk)), marker = 'o', edgecolor = 'none')
	fig.suptitle('Libcarma')
	pos = lnlk.argmax()
	ax[0].axhline(ar_coef_2[pos])
	ax[0].axvline(ar_coef_1[pos])
	ax[0].axhline(np.median(ar_coef_2), color = 'k')
	ax[0].axvline(np.median(ar_coef_1), color = 'k')

	ax[0].axhline(ar_poly[2], color = 'r')
	ax[0].axvline(ar_poly[1], color = 'r')

	print "Libcarma"
	print "Errors"
	print "    Map:",getErrors(theta_exp[pos], theta) 
	print "    Med:",getErrors(np.median(theta_exp, axis = 0), theta)

	show(False)
	fig.canvas.draw()	
	fig.canvas.update()
	fig.canvas.flush_events()

	#pdb.set_trace()
	return newLC

def plotLC(LC, ax = None):

	if ax is None:
		fig, ax = subplots(1,1)
	t = LC.t
	y = LC.y
	yerr = LC.yerr
	
	ax.errorbar(t, y, yerr, marker = 'o', markeredgecolor = 'none', ls = ' ')
	return ax

def buildHash(ar_roots, ma_roots, sigma):

	string = ''.join(map(lambda eks: ''.join(map(str, eks)), (ar_roots, ma_roots, [sigma]))).replace('.','').replace('+','01').replace('-','10').replace('j','11').replace(')','00').replace('(','00').replace('e','11').replace('0','')
	tochar = lambda x, y: chr(48+int(int(x)*int(y)))
	hash_ = ''.join([tochar(string[i], string[i+1]) for i in xrange(len(string)-1)])
	return hash_

class fakeTask:

	def __init__(self):
		pass

def getObjects(p, q, ar_roots, ma_roots, sigma, new = False):

	hash_ = buildHash(ar_roots, ma_roots, sigma)
	if not os.path.isdir('Pickles/'):
		os.mkdir('Pickles/')
		new = True
	if not os.path.isfile('Pickles/Objects-%i-%i-%s.p' % (p, q, hash_)):
		new = True
	if not new:
		print "Loading",hash_
		LC, task, sample = cPickle.load(open('Pickles/Objects-%i-%i-%s.p' % (p,q,hash_),'rb'))
	else:
		print "Generating", hash_
		LC = generateLC(ar_roots, ma_roots, sigma, module = 'libcarma')
		C = start_clock()
		task = fitCARMA(LC, p, q, module = 'libcarma')
		C.terminate()
		print ''
		sample = fitCARMA(LC, p, q, module = 'carmcmc')
		#sample = None
		faketask = fakeTask()
		faketask.Chain = task.Chain
		faketask.LnPosterior = task.LnPosterior
		fakeLC = fakeTask()
		fakeLC.t = LC.t
		fakeLC.x = LC.x
		fakeLC.y = LC.y
		fakeLC.yerr = LC.yerr
		faketask.nsteps = task.nsteps
		package = (fakeLC, faketask, sample)
		cPickle.dump(package,open('Pickles/Objects-%i-%i-%s.p' % (p, q, hash_), 'wb'), 2)
	return (LC, task, sample)

def plotPSD(theta, p, q, ax = None):

	ar_coefs = theta[:p]
	ma_coefs = theta[p:p+1+q]
	sigma = np.ones((1,len(theta)))
	freqs = np.logspace(-5, 5, 500)
	from JacksTools import jools
	l, h, m, f = jools.get_psd(freqs, sigma, ar_coefs, ma_coefs, 95.0)	
	colors = ['#DCA0A0','#EC6060']
	if ax is None:
		fig, ax = subplots(1,1)
		colors = ['#A0A0DC','#6060EC']
		ax.set_xscale('log')
		ax.set_yscale('log')
	ax.fill_between(f, l, h, color = colors[0], alpha = 1.0)
	ax.plot(freqs, m, color = colors[1], alpha = 1.0)
	ax.set_xlim(min(freqs), max(freqs))
	ax.set_ylim(min(m), max(m))

	return ax

def comparePSD(obj1, obj2, p, q):

	t1, ln1 = generateTheta(obj1)
	t2, ln2 = generateTheta(obj2)
	ax = plotPSD(t1, p, q)
	ax = plotPSD(t2, p, q, ax = ax)
	return ax

def main():

	theta, p, q = getTheta(ar_roots, ma_roots, sigma)
	#t, y, yerr = generateLC(ar_roots, ma_roots, sigma, module = 'libcarma')
	LC, task, sample = getObjects(p,q, ar_roots, ma_roots, sigma, new = 1)
	#LC = generateLC(ar_roots, ma_roots, sigma, module = 'libcarma')
	#task = fitCARMA(t, y, yerr, p, q, module = 'libcarma')
	#task = fitCARMA(LC, p, q, module = 'libcarma')
	#sample = fitCARMA(LC, p, q, module = 'carmcmc')

	'''
	walkerPlot(task.Chain, 0, task.LnPosterior, r'$\theta_{0}$')
	walkerPlot(task.Chain, 1, task.LnPosterior, r'$\theta_{1}$')
	walkerPlot(task.Chain, 2, task.LnPosterior, r'$\theta_{2}$')
	walkerPlot(task.Chain, 3, task.LnPosterior, r'$\theta_{3}$')
	walkerPlot(task.Chain, 4, task.LnPosterior, r'$\theta_{4}$')
	walkerPlot(task.Chain, 5, task.LnPosterior, r'$\theta_{5}$')
	errorPlot(task.Chain, task.LnPosterior, theta, r'$Error\ Plot$')
	'''
	trianglePlotCompare(task, sample, theta)
	
	#compareLnLik(task, sample, LC)
	plotLC(LC)
	comparePSD(task, sample, p, q)
	plotPSD(generateTheta(task)[0], p, q)
	Map_lib, Med_lib = Analyze(task, theta)
	Map_car, Med_car = Analyze(sample, theta)

	print "Libcarma Theta Result:"
	print "   Map:", Map_lib 
	print "   Med:", Med_lib 
	print "Carmcmc Theta Result:"
	print "   Map:", Map_car 
	print "   Med:", Med_car 

	print theta

	show()	
	#LC = libcarma_test()
	#ax = plotLC(LC)
	#t, y = carmcmc_test(LC)
	#ax.plot(t, y, marker = 'o', ls = ' ')

	#ar_poly = np.poly(ar_roots)
	#print ar_poly

if __name__ == '__main__':
	main()



