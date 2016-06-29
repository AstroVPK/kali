#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <nlopt.hpp>
#include <vector>
#include <iostream>
#include "Constants.hpp"
#include "binarySMBH.hpp"
//#define DEBUG
//#define DEBUG_KEPLEREQN

#if defined(WRITE)
	#include <stdio.h>
#endif

using namespace std;

double d2r(double degreeVal) {
	return degreeVal*(pi/180.0);
	}

double r2d(double radianVal) {
	return radianVal*(180.0/pi);
	}

double KeplerEqn(const vector<double> &x, vector<double> &grad, void *p2Data) {
	KeplersEqnData *ptr2Data = reinterpret_cast<KeplersEqnData*>(p2Data);
	KeplersEqnData Data = *ptr2Data;
	if (!grad.empty()) {
		grad[0] = 1.0 - Data.ellipticity*cos(x[0]);
		}
	double funcVal = fabs(x[0] - Data.ellipticity*sin(x[0]) - Data.M);
	#ifdef DEBUG_KEPLEREQN
		printf("In Kepler's equation; funcVal = %+4.3e\n",funcVal);
	#endif
	return funcVal;
	}

binarySMBH::binarySMBH() {
	a1 = 1.0e-2*Parsec;
	a2 = 1.0e-2*Parsec;
	m1 = 1.0e7*SolarMass;
	m2 = 1.0e7*SolarMass;
	totalMass = m1 + m2;
	massRatio = m2/m1;
	reducedMass = m1*m2/(m1 + m2);
	ellipticity = 0.0;
	ellipticityFactor = sqrt((1.0 + ellipticity)/(1.0 - ellipticity));
	omega1 = 0.0;
	omega2 = pi;
	inclination = pi/2.0;
	tau = 0.0;
	alpha1 = -0.44;
	alpha2 = -0.44;
	period = sqrt((4.0*piSq*pow(a1, 3.0)*pow(totalMass, 2.0))/(G*pow(m2, 3.0)));
	t = 0.0;
	M = 2.0*pi*(t - tau)/period;
	nlopt::opt opt(nlopt::LN_COBYLA, 1);
	/*vector<double> lb(1), ub(1);
	lb[0] = 0;
	ub[0] = 2.0*pi;
	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(ub);*/
	KeplersEqnData Data;
	Data.ellipticity = ellipticity;
	Data.M = M;
	KeplersEqnData *ptr2Data = &Data;
	opt.set_min_objective(KeplerEqn, ptr2Data);
	opt.set_xtol_rel(1e-16);
	std::vector<double> x(1);
	x[0] = 1.0;
	double minf;
	nlopt::result result = opt.optimize(x, minf);
	E = x[0];
	r1 = a1*(1.0 - ellipticity*cos(E)); // current distance of m1 from COM
	r2 = (m1*r1)/m2; // current distance of m2 from COM
	nu = 2.0*atan(ellipticityFactor*tan(E/2.0)); // current true anomoly of m1
	if (nu < 0.0) {
		nu += 2.0*pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/c;
	beta2 = sqrt(((G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/c;
	radialBeta1 = ((((2.0*pi*a1)/period)*sin(inclination)/sqrt(1.0 - pow(ellipticity, 2.0)))*(cos(nu + omega1) + ellipticity*cos(omega1)))/c;
	radialBeta2 = ((((2.0*pi*a2)/period)*sin(inclination)/sqrt(1.0 - pow(ellipticity, 2.0)))*(cos(nu + omega2) + ellipticity*cos(omega2)))/c;
	dF1 = sqrt(1.0 - pow(beta1, 2.0))/(1.0 - radialBeta1);
	dF2 = sqrt(1.0 - pow(beta2, 2.0))/(1.0 - radialBeta2);
	bF1 = pow((sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1), 3.0 - alpha1);
	bF2 = pow((sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2), 3.0 - alpha2);
	totalFlux = 1.0;
	fracBeamedFlux = 0.0;
	}

binarySMBH::binarySMBH(double a1Val, double a2Val, double m1Val, double m2Val, double ellipticityVal, double omegaVal, double inclinationVal, double tauVal, double alpha1Val, double alpha2Val) {
	a1 = a1Val;
	a2 = a2Val;
	m1 = m1Val;
	m2 = m2Val;
	totalMass = m1 + m2;
	massRatio = m2/m1;
	reducedMass = m1*m2/(m1 + m2);
	ellipticity = ellipticityVal;
	ellipticityFactor = sqrt((1.0 + ellipticity)/(1.0 - ellipticity));
	omega1 = omegaVal;
	omega2 = omega1 + pi;
	inclination = inclinationVal;
	tau = tauVal;
	alpha1 = alpha1Val;
	alpha2 = alpha2Val;
	period = sqrt((4.0*piSq*pow(a1, 3.0)*pow(totalMass, 2.0))/(G*pow(m2, 3.0)));
	t = 0.0;
	M = 2.0*pi*(t - tau)/period;
	nlopt::opt opt(nlopt::LN_COBYLA, 1);
	/*vector<double> lb(1), ub(1);
	lb[0] = 0;
	ub[0] = 2.0*pi;
	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(ub);*/
	KeplersEqnData Data;
	Data.ellipticity = ellipticity;
	Data.M = M;
	KeplersEqnData *ptr2Data = &Data;
	opt.set_min_objective(KeplerEqn, ptr2Data);
	opt.set_xtol_rel(1e-16);
	std::vector<double> x(1);
	x[0] = 1.0;
	double minf;
	nlopt::result result = opt.optimize(x, minf);
	E = x[0];
	r1 = a1*(1.0 - ellipticity*cos(E)); // current distance of m1 from COM
	r2 = (m1*r1)/m2; // current distance of m2 from COM
	nu = 2.0*atan(ellipticityFactor*tan(E/2.0)); // current true anomoly of m1
	if (nu < 0.0) {
		nu += 2.0*pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/c;
	beta2 = sqrt(((G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/c;
	radialBeta1 = ((((2.0*pi*a1)/period)*sin(inclination)/sqrt(1.0 - pow(ellipticity, 2.0)))*(cos(nu + omega1) + ellipticity*cos(omega1)))/c;
	radialBeta2 = ((((2.0*pi*a2)/period)*sin(inclination)/sqrt(1.0 - pow(ellipticity, 2.0)))*(cos(nu + omega2) + ellipticity*cos(omega2)))/c;
	dF1 = sqrt(1.0 - pow(beta1, 2.0))/(1.0 - radialBeta1);
	dF2 = sqrt(1.0 - pow(beta2, 2.0))/(1.0 - radialBeta2);
	bF1 = pow((sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1), 3.0 - alpha1);
	bF2 = pow((sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2), 3.0 - alpha2);
	totalFlux = 1.0;
	fracBeamedFlux = 0.0;
	}

void binarySMBH::operator()(double epoch) {
	setEpoch(epoch);
	M = 2.0*pi*(t - tau)/period;
	nlopt::opt opt(nlopt::LN_COBYLA, 1);
	/*vector<double> lb(1), ub(1);
	lb[0] = 0;
	ub[0] = 2.0*pi;
	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(ub);*/
	KeplersEqnData Data;
	Data.ellipticity = ellipticity;
	Data.M = M;
	KeplersEqnData *ptr2Data = &Data;
	opt.set_min_objective(KeplerEqn, ptr2Data);
	opt.set_xtol_rel(1e-16);
	std::vector<double> x(1);
	x[0] = 1.0;
	double minf;
	nlopt::result result = opt.optimize(x, minf);
	E = x[0];
	r1 = a1*(1.0 - ellipticity*cos(E)); // current distance of m1 from COM
	r2 = (m1*r1)/m2; // current distance of m2 from COM
	nu = 2.0*atan(ellipticityFactor*tan(E/2.0)); // current true anomoly of m1
	if (nu < 0.0) {
		nu += 2.0*pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/c;
	beta2 = sqrt(((G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/c;
	radialBeta1 = ((((2.0*pi*a1)/period)*sin(inclination)/sqrt(1.0 - pow(ellipticity, 2.0)))*(cos(nu + omega1) + ellipticity*cos(omega1)))/c;
	radialBeta2 = ((((2.0*pi*a2)/period)*sin(inclination)/sqrt(1.0 - pow(ellipticity, 2.0)))*(cos(nu + omega2) + ellipticity*cos(omega2)))/c;
	dF1 = sqrt(1.0 - pow(beta1, 2.0))/(1.0 - radialBeta1);
	dF2 = sqrt(1.0 - pow(beta2, 2.0))/(1.0 - radialBeta2);
	bF1 = pow((sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1), 3.0 - alpha1);
	bF2 = pow((sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2), 3.0 - alpha2);
	}

int binarySMBH::checkBinarySMBHParams(double *Theta) {
	if ((Theta[0] > 0.0) and (Theta[1] > 0.0) and (Theta[2] > 0.0) and (Theta[3] > 0.0) and (Theta[4] >= 0.0) and (Theta[4] < 1.0) and (Theta[5] >= 0.0) and (Theta[5] < twoPi) and (Theta[6] >= 0.0) and (Theta[6] < pi) and (Theta[7] >= 0.0) and (Theta[8] > 0.0) and (Theta[9] > 0.0) and (Theta[9] <= 1.0)) {
		return 1;
		} else {
		return 0;
		}
	}

void binarySMBH::setBinarySMBHParams(double *Theta) {
	a1 = Parsec*Theta[0];
	a2 = Parsec*Theta[1];
	m1 = 1.0e6*SolarMass*Theta[2];
	m2 = 1.0e6*SolarMass*Theta[3];
	totalMass = m1 + m2;
	massRatio = m2/m1;
	reducedMass = m1*m2/(m1 + m2);
	ellipticity = Theta[4];
	ellipticityFactor = sqrt((1.0 + ellipticity)/(1.0 - ellipticity));
	omega1 = Theta[5];
	omega2 = omega1 + pi;
	inclination = Theta[6];
	tau = Theta[7];
	alpha1 = -0.44;
	alpha2 = -0.44;
	period = sqrt((4.0*piSq*pow(a1, 3.0)*pow(totalMass, 2.0))/(G*pow(m2, 3.0)));
	t = 0.0;
	M = 2.0*pi*(t - tau)/period;
	nlopt::opt opt(nlopt::LN_COBYLA, 1);
	/*vector<double> lb(1), ub(1);
	lb[0] = 0;
	ub[0] = 2.0*pi;
	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(ub);*/
	KeplersEqnData Data;
	Data.ellipticity = ellipticity;
	Data.M = M;
	KeplersEqnData *ptr2Data = &Data;
	opt.set_min_objective(KeplerEqn, ptr2Data);
	opt.set_xtol_rel(1e-16);
	std::vector<double> x(1);
	x[0] = 1.0;
	double minf;
	nlopt::result result = opt.optimize(x, minf);
	E = x[0];
	r1 = a1*(1.0 - ellipticity*cos(E)); // current distance of m1 from COM
	r2 = (m1*r1)/m2; // current distance of m2 from COM
	nu = 2.0*atan(ellipticityFactor*tan(E/2.0)); // current true anomoly of m1
	if (nu < 0.0) {
		nu += 2.0*pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/c;
	beta2 = sqrt(((G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/c;
	radialBeta1 = ((((2.0*pi*a1)/period)*sin(inclination)/sqrt(1.0 - pow(ellipticity, 2.0)))*(cos(nu + omega1) + ellipticity*cos(omega1)))/c;
	radialBeta2 = ((((2.0*pi*a2)/period)*sin(inclination)/sqrt(1.0 - pow(ellipticity, 2.0)))*(cos(nu + omega2) + ellipticity*cos(omega2)))/c;
	dF1 = sqrt(1.0 - pow(beta1, 2.0))/(1.0 - radialBeta1);
	dF2 = sqrt(1.0 - pow(beta2, 2.0))/(1.0 - radialBeta2);
	bF1 = pow((sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1), 3.0 - alpha1);
	bF2 = pow((sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2), 3.0 - alpha2);
	totalFlux = Theta[8];
	fracBeamedFlux = Theta[9];
	}

double binarySMBH::getEpoch() {return t;}
void binarySMBH::setEpoch(double epoch) {t = epoch;}
double binarySMBH::getPeriod() {return period;}
double binarySMBH::getA1() {return a1;}
double binarySMBH::getA2() {return a2;}
double binarySMBH::getEllipticity() {return ellipticity;}
double binarySMBH::getR1() {return r1;}
double binarySMBH::getR2() {return r2;}
double binarySMBH::getTheta1() {return theta1;}
double binarySMBH::getTheta2() {return theta2;}
double binarySMBH::getBeta1() {return beta1;}
double binarySMBH::getBeta2() {return beta2;}
double binarySMBH::getRadialBeta1() {return radialBeta1;}
double binarySMBH::getRadialBeta2() {return radialBeta2;}
double binarySMBH::getDopplerFactor1() {return dF1;}
double binarySMBH::getDopplerFactor2() {return dF2;}
double binarySMBH::getBeamingFactor1() {return bF1;}
double binarySMBH::getBeamingFactor2() {return bF2;}

double binarySMBH::aH(double sigmaStars) {
	double aHVal = (G*reducedMass)/(4.0*pow(sigmaStars, 2.0));
	return aHVal;
	}

double binarySMBH::aGW(double sigmaStars, double rhoStars, double H) {
	double aGWVal = pow((64.0*pow(G*reducedMass, 2.0)*totalMass*sigmaStars)/(5.0*H*pow(c, 5.0)*rhoStars), 0.2);
	return aGWVal;
	}

double binarySMBH::durationInHardState(double sigmaStars, double rhoStars, double H) {
	double durationInHardStateVal = (sigmaStars/(H*G*rhoStars*aGW(sigmaStars, rhoStars, H)));
	return durationInHardStateVal;
	}

double binarySMBH::ejectedMass(double sigmaStars, double rhoStars, double H) {
	double ejectedMassVal = totalMass*log(aH(sigmaStars)/aGW(sigmaStars, rhoStars, H));
	return ejectedMassVal;
	}

void binarySMBH::print() {
	cout << "a1: " << a1/Parsec << " (pc)" << endl;
	cout << "a2: " << a2/Parsec << " (pc)" << endl;
	cout << "m1: " << m1/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << "m2: " << m2/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << "Total Mass: " << totalMass/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << "Mass Ratio: " << massRatio << endl;
	cout << "Reduced Mass: " << reducedMass/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << "Period: " << period/Day << " (day) == " << period/Year  << " (year)" << endl;
	cout << "Eccentricity: " << ellipticity << endl;
	cout << "Longitude of the ascending node (mass 1): " << r2d(omega1) << " (degree)" << endl;
	cout << "Inclination: " << r2d(inclination) << " (degree)" << endl;
	cout << "Time of Periastron: " << tau/Day << " (day)" << endl;
	cout << "Total Flux: " << totalFlux << endl;
	cout << "Beamed Flux Fraction: " << fracBeamedFlux << endl;
	}

void binarySMBH::simulateSystem(LnLikeData *ptr2Data) {
	LnLikeData Data = *ptr2Data;

	int numCadences = Data.numCadences;
	double *t = Data.t;
	double *x = Data.x;

	for (int i = 0; i < numCadences; ++i) {
		(*this)(t[i]*Day);
		x[i] = totalFlux*fracBeamedFlux*getBeamingFactor2() + totalFlux*(1.0 - fracBeamedFlux);
		}
	}

void binarySMBH::observeNoise(LnLikeData *ptr2Data, unsigned int noiseSeed, double* noiseRand) {
	LnLikeData Data = *ptr2Data;

	int numCadences = Data.numCadences;
	double fracNoiseToSignal = Data.fracNoiseToSignal;
	double *t = Data.t;
	double *x = Data.x;
	double *y = Data.y;
	double *yerr = Data.yerr;
	double *mask = Data.mask;

	double noiseLvl = 0.0;
	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	VSLStreamStatePtr noiseStream __attribute__((aligned(64)));
	vslNewStream(&noiseStream, VSL_BRNG_SFMT19937, noiseSeed);
	for (int i = 0; i < numCadences; i++) {
		noiseLvl = fracNoiseToSignal*x[i];
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, noiseStream, 1, &noiseRand[i], 0.0, noiseLvl);
		y[i] = x[i] + noiseRand[i];
		yerr[i] = noiseLvl;
		}
	}