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
//#define DEBUG_SIMULATESYSTEM
//#ifdef DEBUG_CHECKBINARYSMBHPARAMS

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
	rPer = 0.0;
	m1 = 0.0;
	m2 = 0.0;
	rS1 = 0.0;
	rS2 = 0.0;
	totalMass = 0.0;
	massRatio = NAN;
	reducedMass = 0.0;
	ellipticity = 0.0;
	ellipticityFactor = 1.0;
	a1 = 0.0;
	a2 = 0.0;
	omega1 = 0.0;
	omega2 = pi;
	inclination = pi/2.0;
	tau = 0.0;
	alpha1 = -0.44;
	alpha2 = -0.44;
	period = 0.0;
	t = 0.0;
	M = 0.0;
	E = 0.0;
	r1 = 0.0; // current distance of m1 from COM
	r2 = 0.0; // current distance of m2 from COM
	nu = 0.0; // current true anomoly of m1
	if (nu < 0.0) {
		nu += 2.0*pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = 0.0;
	beta2 = 0.0;
	radialBeta1 = 0.0;
	radialBeta2 = 0.0;
	dF1 = 0.0;
	dF2 = 0.0;
	bF1 = 0.0;
	bF2 = 0.0;
	totalFlux = 1.0;
	fracBeamedFlux = 0.0;
	}

binarySMBH::binarySMBH(double rPerVal, double m1Val, double m2Val, double ellipticityVal, double omegaVal, double inclinationVal, double tauVal, double alpha1Val, double alpha2Val) {
	rPer = rPerVal;
	m1 = m1Val;
	m2 = m2Val;
	rS1 = 2.0*G*m1/pow(c, 2.0);
	rS2 = 2.0*G*m2/pow(c, 2.0);
	totalMass = m1 + m2;
	massRatio = m2/m1;
	reducedMass = m1*m2/(m1 + m2);
	ellipticity = ellipticityVal;
	ellipticityFactor = sqrt((1.0 + ellipticity)/(1.0 - ellipticity));
	a1 = (m2*rPer)/(totalMass*(1.0 - ellipticity));
	a2 = (m1*rPer)/(totalMass*(1.0 - ellipticity));
	omega1 = omegaVal;
	omega2 = omega1 + pi;
	inclination = inclinationVal;
	tau = tauVal;
	alpha1 = alpha1Val;
	alpha2 = alpha2Val;
	period = twoPi*sqrt(pow(a1 + a2, 3.0)/(G*totalMass));
	t = 0.0;
	M = 2.0*pi*(t - tau)/period;
	nlopt::opt opt(nlopt::LN_COBYLA, 1);
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

int binarySMBH::checkBinarySMBHParams(double *ThetaIn) {
	double rPerVal = Parsec*ThetaIn[0];
	double m1Val = SolarMass*1.0e6*ThetaIn[1];
	double m2Val = SolarMass*1.0e6*ThetaIn[2];
	double rS1Val = (2.0*G*m1Val)/(pow(c, 2.0));
	double rS2Val = (2.0*G*m2Val)/(pow(c, 2.0));
	double ellipticityVal = ThetaIn[3];
	double omega1Val = d2r(ThetaIn[4]);
	double inclinationVal = d2r(ThetaIn[5]);
	double tauVal = ThetaIn[6]*Day;
	double totalFluxVal = ThetaIn[7];
	double fracBeamedFluxVal = ThetaIn[8];
	if ((m1Val > 0.0) and (m2Val > 0.0) and (ellipticityVal >= 0.0) and (ellipticity < 1.0) and (omega1Val >= 0.0) and (omega1Val < twoPi) and (inclinationVal >= 0.0) and (inclinationVal <= pi) and (tauVal >= 0.0) and (totalFluxVal > 0.0) and (fracBeamedFluxVal > 0.0) and (fracBeamedFluxVal <= 1.0)) {
		if (rPerVal > (10.0*(rS1Val + rS2Val))) {
			return 1;
			} else {
			printf("rPer = %+4.3e pc < %+4.3e pc i.e. 10.0*(rs1 + rs2); Binaries approach too close!\n", rPerVal/Parsec, (10.0*(rS1Val + rS2Val))/Parsec);
			return 0;
			}
		} else {
		return 0;
		}
	}

void binarySMBH::setBinarySMBH(double *Theta) {
	rPer = Parsec*Theta[0];
	m1 = SolarMass*1.0e6*Theta[1];
	m2 = SolarMass*1.0e6*Theta[2];
	rS1 = (2.0*G*m1)/(pow(c, 2.0));
	rS2 = (2.0*G*m2)/(pow(c, 2.0));
	totalMass = m1 + m2;
	massRatio = m2/m1;
	reducedMass = m1*m2/(m1 + m2);
	ellipticity = Theta[3];
	ellipticityFactor = sqrt((1.0 + ellipticity)/(1.0 - ellipticity));
	a1 = (m2*rPer)/(totalMass*(1.0 - ellipticity));
	a2 = (m1*rPer)/(totalMass*(1.0 - ellipticity));
	omega1 = d2r(Theta[4]);
	omega2 = omega1 + pi;
	inclination = d2r(Theta[5]);
	tau = Theta[6]*Day;
	alpha1 = -0.44;
	alpha2 = -0.44;
	period = twoPi*sqrt(pow(a1 + a2, 3.0)/(G*totalMass));
	t = 0.0;
	M = 2.0*pi*(t - tau)/period;
	nlopt::opt opt(nlopt::LN_COBYLA, 1);
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
	totalFlux = Theta[7];
	fracBeamedFlux = Theta[8];
	}

double binarySMBH::getEpoch() {return t/Day;}
void binarySMBH::setEpoch(double epoch) {t = epoch;}
double binarySMBH::getPeriod() {return period/Day;}
double binarySMBH::getA1() {return a1/Parsec;}
double binarySMBH::getA2() {return a2/Parsec;}
double binarySMBH::getM1() {return m1/(SolarMass*1.0e6);}
double binarySMBH::getM2() {return m2/(SolarMass*1.0e6);}
double binarySMBH::getRS1() {return rS1/Parsec;}
double binarySMBH::getRS2() {return rS2/Parsec;}
double binarySMBH::getEllipticity() {return ellipticity;}
double binarySMBH::getMeanAnomoly() {return r2d(M);}
double binarySMBH::getEccentricAnomoly() {return r2d(E);}
double binarySMBH::getTrueAnomoly() {return r2d(nu);}
double binarySMBH::getR1() {return r1/Parsec;}
double binarySMBH::getR2() {return r2/Parsec;}
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
	cout.precision(5);
	cout << scientific << "                           rPer1 + rPer2: " << rPer/Parsec << " (pc)" << endl;
	cout << scientific << "                                      a1: " << a1/Parsec << " (pc)" << endl;
	cout << scientific << "                                      a2: " << a2/Parsec << " (pc)" << endl;
	cout << scientific << "                                      m1: " << m1/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                                      m2: " << m2/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                                     rS1: " << rS1/Parsec << " (pc)" << endl;
	cout << scientific << "                                     rS2: " << rS2/Parsec << " (pc)" << endl;
	cout << scientific << "                              Total Mass: " << totalMass/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                              Mass Ratio: " << massRatio << endl;
	cout << scientific << "                            Reduced Mass: " << reducedMass/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                                  Period: " << period/Day << " (day) == " << period/Year  << " (year)" << endl;
	cout << scientific << "                            Eccentricity: " << ellipticity << endl;
	cout << scientific << "Longitude of the ascending node (mass 1): " << r2d(omega1) << " (degree)" << endl;
	cout << scientific << "                             Inclination: " << r2d(inclination) << " (degree)" << endl;
	cout << scientific << "                      Time of Periastron: " << tau/Day << " (day)" << endl;
	cout << scientific << "                              Total Flux: " << totalFlux << endl;
	cout << scientific << "                    Beamed Flux Fraction: " << fracBeamedFlux << endl;
	}

void binarySMBH::simulateSystem(LnLikeData *ptr2Data) {
	LnLikeData Data = *ptr2Data;

	int numCadences = Data.numCadences;
	double *t = Data.t;
	double *x = Data.x;

	for (int i = 0; i < numCadences; ++i) {
		(*this)(t[i]*Day);
		#ifdef DEBUG_SIMULATESYSTEM
			printf("t[%d]: %+4.3e\n", i, getEpoch());
			printf("a2: %+4.3e\n", getA2());
			printf("r2: %+4.3e\n", getR2());
			printf("theta2: %+4.3e\n", getTheta2());
			printf("M: %+4.3e\n", getMeanAnomoly());
			printf("E: %+4.3e\n", getEccentricAnomoly());
			printf("nu: %+4.3e\n", getTrueAnomoly());
			printf("beta2: %+4.3e\n", getBeta2());
			printf("rbeta2: %+4.3e\n", getRadialBeta2());
			printf("dF2: %+4.3e\n", getDopplerFactor2());
			printf("bF2: %+4.3e\n", getBeamingFactor2());
		#endif
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