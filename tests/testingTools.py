'''
Library of functions to acilitate libcarma testing
everything should work in terms of Rho, Theta, p and q
or arrays of rho, theta, p and q
'''
import sys, os, time, pdb
import numpy as np
import libcarma, carmcmc
from matplotlib.pyplot import subplots, show, figure
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def slowdown():
	if np.random.random() > 0.95:
		time.sleep(np.absolute(np.random.randn()*5))
		print ":("

def nBins(data):
	'''find ideal number of bins for histogramming data'''
	q75, q25 = np.percentile(data, [75, 25]) #start by finding number of bins
	IQR = q75 - q25
	nbins = int((max(data) - min(data))/(2*IQR*len(data)**(-1.0/3)))
	return nbins

def Rho(AR_ROOTS, MA_ROOTS, B_Q):
	'''Convert Parameters to a array of roots'''
	rho = np.concatenate((AR_ROOTS, MA_ROOTS, [B_Q]))
	return rho	

def Theta(AR_ROOTS, MA_ROOTS, B_Q):
	'''Theta from inputs'''
	theta = libcarma.coeffs(len(AR_ROOTS), len(MA_ROOTS), Rho(AR_ROOTS, MA_ROOTS, B_Q))
	return theta

def CarmaSampleTheta(sample):
	'''get list of Theta from CarmaSample'''
	ma_coefs = sample.get_samples('ma_coefs')
	ar_coefs = sample.get_samples('ar_coefs')[:,1:]
	sigma = sample.get_samples('sigma')
	ma_poly = (sigma*ma_coefs)#[::-1]
	theta = np.concatenate((ar_coefs, ma_poly), axis = -1)
	return theta

def CarmaSampleRho(sample):
	'''get list of roots from CarmaSample'''
	theta = CarmaSampleTheta(sample)
	rho = np.array([libcarma.roots(sample.p, sample.q, theta[i]) for i in xrange(len(theta))])
	return rho	

def LibcarmaTheta(task):
	Theta = np.array([task.Chain[i,:,task.nsteps/4*3:].flatten() for i in xrange(task.Chain.shape[0])]).T
	return Theta

def LibcarmaRho(task):
	theta = LibcarmaTheta(task)
	print task.nsteps, task.nwalkers, theta.shape
	rho = np.array([libcarma.roots(task.p, task.q, theta[i]) for i in xrange(task.nwalkers*task.nsteps/4*1)])
	return rho

def createLC(theta, p, q, dt, T, nwalkers = 100, nsteps = 500):
	'''Create a new light curve with given parameters'''
	newTask = libcarma.basicTask(p, q, nwalkers = nwalkers, nsteps = nsteps)
	newTask.set(dt, theta)
	newLC = newTask.simulate(T)
	newTask.observe(newLC)

	return newLC

def fitCARMA(LC, p, q, module = 'libcarma', *args, **kwargs):
	'''fit a light curve to a carma process'''
	if module == 'libcarma':
		guess = np.zeros((p+q+1)) + libcarma.coeffs(p,q,np.concatenate((np.random.random(p), np.random.random(q)*1e-7+1e-7, np.random.random(1)*1e-7+1e-7))
		newTask = libcarma.basicTask(p, q, *args, **kwargs)
		newTask.fit(LC, guess)
		return newTask
	elif module == 'carmcmc':
		t, y, yerr = LC.t, LC.y, LC.yerr
		model = carmcmc.CarmaModel(t, y, yerr, p = p, q = q)
		sample = model.run_mcmc(*args, **kwargs)
		return sample


def DistanceCompare(theta1, lnlk1, theta2, lnlk2, theta):

	meds = [np.median(theta1, axis = 0), np.median(theta2,axis = 0)]
	maps = [theta1[lnlk1.argmax()], theta2[lnlk2.argmax()]]
	f = lambda x: np.sum(((x - theta)/theta)**2)
	meds = map(f, meds)
	maps = map(f, maps)

	title = '      Libcarma | Carmcmc'
	line1 = 'map: '+"%.6f" % maps[0]+' | '+"%.6f" % maps[1]
	line2 = 'med: '+"%.6f" % meds[0]+' | '+"%.6f" % meds[1]
	return '\n'.join((title, line1, line2))

def trianglePlotCompare(theta1, lnlk1, theta2, lnlk2, theta = None):

	mock = bool(theta is not None)
	pos1 = lnlk1.argmax()
	pos2 = lnlk2.argmax()

	m, n = theta1.shape
	fig = figure(figsize = (n*5, n*5), dpi = 100)
	for i in xrange(n):
		slowdown()
		for j in xrange(i+1):
			 if i == j:
				  nbins = nBins(np.concatenate((theta1[:,i], theta2[:,i]), axis = 0))
				  ax = fig.add_subplot(n, n, i*n+j+1)
				  ax.set_title(r'$\theta_{%i}$' % (j + 1))
				  num, bins, patches = ax.hist([theta1[:,i], theta2[:,i]], bins = nbins, histtype = 'stepfilled', normed = True, color = ['#A0A0DC','#DCA0A0'], alpha = 1.0, stacked = True)
				  if mock: ax.axvline(theta[i], color = 'g')
				  ax.axvline(theta1[pos1,i], color = 'b')
				  ax.axvline(theta2[pos2,i], color = 'r')
				  ax.autoscale_view(False, False, False)
				  ax.autoscale(False, False, False)
				  ax.set_xlim(min(bins), max(bins))
			 else:
				  yAxis = None if j == 0 else fig.axes[i*(i+1)/2]
				  xAxis = fig.axes[(j+1)*(j+2)/2-1]
				  ax = fig.add_subplot(n, n, n*i+j+1, sharex = xAxis, sharey = yAxis)
				  ax.scatter(theta1[:,j], theta1[:,i], c = lnlk1, marker = 'o', edgecolor = 'none', alpha = 0.5, cmap = 'cool')
				  ax.scatter(theta2[:,j], theta2[:,i], c = lnlk2, marker = 'o', edgecolor = 'none', alpha = 0.5, cmap = 'autumn')
				  if mock:
						ax.axvline(theta[j], color = 'g')
						ax.axhline(theta[i], color = 'g')
				  ax.axhline(theta1[pos1,i], color = 'b')
				  ax.axvline(theta1[pos1,j], color = 'b')
				  ax.axhline(theta2[pos2,i], color = 'r')
				  ax.axvline(theta2[pos2,j], color = 'r')
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

def abslog(x):
	return np.log10(np.absolute(x))

def loadLC(self, name, path = None):
	from JacksTools import jio, jools
	fname = ''
	while not os.path.isfile(fname): 
		fname = "LightCurveSDSS_%i.csv" % np.random.randint(1,56)
	fname = "LightCurveSDSS_39.csv"# % np.random.randint(1,56)
	data = jio.load(fname, headed = True, delimiter = ',')
	flux, err = jools.luptitude_to_flux(data['calMag_g'], data['calMagErr_g'], 'g')
	#flux = data['psfMag_g']
	#err = data['psfMagErr_g']
	z = 1.074
	t = jools.time_to_restFrame(data['mjd_g'], z)
	t = t - min(t)
	#flux = jools.flux_to_lum(flux, z)
	#err = jools.flux_to_lum(err, z)
	
	self._p = 0
	self._q = 0
	self._computedCadenceNum = -1
	self._tolIR = 1.0e-3
	self._fracIntrinsicVar = 0.0
	self._fracNoiseToSignal = 0.0
	self._maxSigma = 2.0
	self._minTimescale = 2.0
	self._maxTimescale = 0.5
	if path is None:
		path = os.environ['PWD']
	#open kepler k2 lc fits file
	self._numCadences = len(data)
	self._dt = np.nanmin(t[1:] - t[:-1]) ## Increment between epochs.
	print self._dt
	self._T = t[-1] - t[0] ## Total duration of the light curve.
	print self._T
	self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
	self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
	self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
	self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
	self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
	self.mask = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E']) ## Numpy array of mask values.
	for i in xrange(self.numCadences):
		self.cadence[i] = i
		if 1:
			self.t[i] = t[i]
			self.y[i] = flux[i]
			self.yerr[i] = err[i]
			self.mask[i] = 1.0
		else:
			if np.isnan(t[i]) == False:
				self.t[i] = t[i]
			else:
				self.t[i] = self.t[i - 1] + self._dt
		self.mask[i] = 0.0
	#self._dt = np.nanmedian(self.t[1:] - self.t[:-1]) ## Increment between epochs.
	self._p = 0
	self._q = 0
	self.XSim = np.require(np.zeros(self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## State of light curve at last timestamp
	self.PSim = np.require(np.zeros(self._p*self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
	self.XComp = np.require(np.zeros(self._p*self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
	self.PComp = np.require(np.zeros(self._p*self._p), requirements=['F', 'A', 'W', 'O', 'E']) ## Uncertainty in state of light curve at last timestamp.
	self._name = str(fname.split('.')[0]) ## The name of the light curve (usually the object's name).
	self._band = str('g') ## The name of the photometric band (eg. HSC-I or SDSS-g etc..).
	self._xunit = r'$d$' ## Unit in which time is measured (eg. s, sec, seconds etc...).
	self._yunit = r'$who the f**** knows?$' ## Unit in which the flux is measured (eg Wm^{-2} etc...).

	
libcarma.basicLC.read = loadLC



def main():

	dt = 0.4
	T = 1079
	numCadences = int(T/dt)

	maxSigma = 2.0
	minTimescale = 5.0e-1
	maxTimescale = 2.0

	AR_ROOTS = [-0.0333642081+0.0254j, -0.033642081-0.0254j]
	MA_ROOTS = [-3.833333]
	B_Q = 2.0e-9
	p = len(AR_ROOTS)
	q = len(MA_ROOTS)


	theta = Theta(AR_ROOTS, MA_ROOTS, B_Q)
	strings = []
	tims = []
	for i in xrange(1):
		#LC = createLC(theta, p, q, dt, T)
		LC = libcarma.basicLC(1, supplied = True)

		fig, ax = subplots(1,1)
		ax.errorbar(LC.t, LC.y, LC.yerr, ls = ' ', marker = '.', markeredgecolor = 'none', color = '#D95F02')

		#LC.sampler = 'matchSampler'
		#LC.sample(timestamps=LC1.t)

		LC.minTimescale = 0.5
		LC.maxTimescale = 2.0
		T0 = time.time()
		task = fitCARMA(LC, p, q, module = 'libcarma', nwalkers = 100, nsteps = 1000)
		T1 = time.time() - T0
		print "Time1:", T1
		T0 = time.time()
		###!!!
		#pos = np.in1d(LC.t, LC1.t)
		#LC.t = LC.t[pos]
		#LC.y = LC.y[pos]
		#LC.yerr = LC.yerr[pos]
		###!!!
		sample = fitCARMA(LC, p, q, 'carmcmc', 25*1000, nburnin = 75*1000)
		T2 = time.time() - T0
		print "Time2:", T2
		tims.append(1.0*T2/T1)
		print "Theta:", theta

		#theta1 = LibcarmaTheta(task)
		theta1 = np.log10(np.absolute(1.0/LibcarmaRho(task).real))
		lnlk1 = task.LnPosterior[:,3* task.nsteps/4:].flatten()
		#theta2 = CarmaSampleTheta(sample)
		theta2 = np.log10(np.absolute(1.0/CarmaSampleRho(sample).real))
		lnlk2 = sample.get_samples('logpost')[:,0]
		#strings.append(DistanceCompare(theta1, lnlk1, theta2, lnlk2, np.log10(np.absolute(1.0/libcarma.roots(p, q, theta).real))))
		#trianglePlotCompare(theta1, lnlk1, theta2, lnlk2, np.log10(np.absolute(1.0/libcarma.roots(p, q, theta).real)))
		try:
			trianglePlotCompare(theta1, lnlk1, theta2, lnlk2, np.log10(np.ones(theta1.shape[1])))
		except:
			pass
		show(False)
		fig.canvas.draw()
		fig.canvas.update()
		fig.canvas.flush_events()
		time.sleep(0.5)
	#print "Rho:",abslog(1.0/theta)
	#for i,s in enumerate(strings):
	#	print i
	#	print s
	
	show(False)
	print np.mean(tims)
	print np.std(tims)

	pdb.set_trace()
if __name__ == '__main__':
	main()
