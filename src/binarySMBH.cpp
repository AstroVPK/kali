#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <nlopt.hpp>
#include <limits>
#include <vector>
#include <iostream>
#include <stdio.h>
#include "Constants.hpp"
#include "binarySMBH.hpp"

//#define DEBUG_KEPLEREQN
//#define DEBUG_SIMULATESYSTEM
//#define DEBUG_CHECKBINARYSMBHPARAMS
//#define DEBUG_CALCLNPOSTERIOR

//#if defined(DEBUG_KEPLEREQN) || defined(DEBUG_SIMULATESYSTEM) || defined(DEBUG_CHECKBINARYSMBHPARAMS) || defined(DEBUG_CALCLNPOSTERIOR)
//	#include <stdio.h>
//#endif

int lenThetaAlso = 8;

using namespace std;

double calcLnPrior(const vector<double> &x, vector<double>& grad, void *p2Args) {
	/*! Used for computing good regions */
	if (!grad.empty()) {
		#pragma omp simd
		for (int i = 0; i < x.size(); ++i) {
			grad[i] = 0.0;
			}
		}
	int threadNum = omp_get_thread_num();
	LnLikeArgs *ptr2Args = reinterpret_cast<LnLikeArgs*>(p2Args);
	LnLikeArgs Args = *ptr2Args;
	binarySMBH *Systems = Args.Systems;
	double LnPrior = 0.0;

	if (Systems[threadNum].checkBinarySMBHParams(const_cast<double*>(&x[0])) == 1) {
		LnPrior = 0.0;
		} else {
		LnPrior = -infiniteVal;
		}
	return LnPrior;
	}

double calcLnPrior(double *walkerPos, void *func_args) {
	/*! Used for computing good regions */
	int threadNum = omp_get_thread_num();
	LnLikeArgs *ptr2Args = reinterpret_cast<LnLikeArgs*>(func_args);
	LnLikeArgs Args = *ptr2Args;
	binarySMBH* Systems = Args.Systems;
	double LnPrior = 0.0;
	if (Systems[threadNum].checkBinarySMBHParams(walkerPos) == 1) {
		LnPrior = 0.0;
		} else {
		LnPrior = -infiniteVal;
		}
	return LnPrior;
	}

double calcLnPosterior(const vector<double> &x, vector<double>& grad, void *p2Args) {
	if (!grad.empty()) {
		#pragma omp simd
		for (int i = 0; i < x.size(); ++i) {
			grad[i] = 0.0;
			}
		}
	int threadNum = omp_get_thread_num();
	LnLikeArgs *ptr2Args = reinterpret_cast<LnLikeArgs*>(p2Args);
	LnLikeArgs Args = *ptr2Args;
	LnLikeData *ptr2Data = Args.Data;
	binarySMBH *Systems = Args.Systems;
	double LnPosterior = 0.0, old_dt = 0.0;
	if (Systems[threadNum].checkBinarySMBHParams(const_cast<double*>(&x[0])) == 1) {
		Systems[threadNum].setBinarySMBH(const_cast<double*>(&x[0]));
		#ifdef DEBUG_CALCLNPOSTERIOR
		#pragma omp critical
		{
			printf("calcLnPosterior - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < lenThetaAlso; dimNum++) {
				printf("%+17.16e ", x[dimNum]);
				}
			printf("\n");
			fflush(0);
		}
		#endif
		LnPosterior = Systems[threadNum].computeLnLikelihood(ptr2Data) + Systems[threadNum].computeLnPrior(ptr2Data);
		#ifdef DEBUG_CALCLNPOSTERIOR
		#pragma omp critical
		{
			printf("calcLnPosterior - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < lenThetaAlso; dimNum++) {
				printf("%+17.16e ", x[dimNum]);
				}
			printf("LnLike: %f\n",LnPosterior);
			printf("\n");
		}
		#endif
		} else {
		LnPosterior = -infiniteVal;
		}
	return LnPosterior;
	}

double calcLnPosterior(double *walkerPos, void *func_args) {
	int threadNum = omp_get_thread_num();
	LnLikeArgs *ptr2Args = reinterpret_cast<LnLikeArgs*>(func_args);
	LnLikeArgs Args = *ptr2Args;
	LnLikeData *Data = Args.Data;
	binarySMBH *Systems = Args.Systems;
	LnLikeData *ptr2Data = Data;
	double LnPosterior = 0.0, old_dt = 0.0;
	if (Systems[threadNum].checkBinarySMBHParams(walkerPos) == 1) {
		Systems[threadNum].setBinarySMBH(walkerPos);
		#ifdef DEBUG_CALCLNPOSTERIOR
		#pragma omp critical
		{
			printf("calcLnPosterior - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < lenThetaAlso; dimNum++) {
				printf("%+17.16e ", walkerPos[dimNum]);
				}
			printf("\n");
			fflush(0);
		}
		#endif
		LnPosterior = Systems[threadNum].computeLnLikelihood(ptr2Data) + Systems[threadNum].computeLnPrior(ptr2Data);
		#ifdef DEBUG_CALCLNPOSTERIOR
		#pragma omp critical
		{
			printf("calcLnPosterior - threadNum: %d; walkerPos: ",threadNum);
			for (int dimNum = 0; dimNum < lenThetaAlso; dimNum++) {
				printf("%+17.16e ", walkerPos[dimNum]);
				}
			printf("LnLike: %f\n",LnPosterior);
			printf("\n");
		}
		#endif
		} else {
		LnPosterior = -infiniteVal;
		}
	return LnPosterior;
	}

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
		grad[0] = 1.0 - Data.eccentricity*cos(x[0]);
		}
	double funcVal = fabs(x[0] - Data.eccentricity*sin(x[0]) - Data.M);
	#ifdef DEBUG_KEPLEREQN
		printf("In Kepler's equation; funcVal = %+4.3e\n",funcVal);
	#endif
	return funcVal;
	}

binarySMBH::binarySMBH() {
	rPeriTot = 0.0;
	m1 = 0.0;
	m2 = 0.0;
	rS1 = 0.0;
	rS2 = 0.0;
	totalMass = 0.0;
	massRatio = NAN;
	reducedMass = 0.0;
	eccentricity = 0.0;
	eccentricityFactor = 1.0;
	a1 = 0.0;
	a2 = 0.0;
	rPeri1 = 0.0;
	rPeri2 = 0.0;
	rApo1 = 0.0;
	rApo2 = 0.0;
	omega1 = 0.0;
	omega2 = pi;
	inclination = pi/2.0;
	tau = 0.0;
	alpha1 = -0.44;
	alpha2 = -0.44;
	period = 0.0;
	epoch = 0.0;
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
	_radialBetaFactor1 = 0.0;
	_radialBetaFactor2 = 0.0;
	radialBeta1 = 0.0;
	radialBeta2 = 0.0;
	dF1 = 0.0;
	dF2 = 0.0;
	bF1 = 0.0;
	bF2 = 0.0;
	totalFlux = 1.0;
	//fracBeamedFlux = 0.0;
	}

binarySMBH::binarySMBH(double a1Val, double a2Val, double periodVal, double eccentricityVal, double omegaVal, double inclinationVal, double tauVal, double alpha1Val, double alpha2Val) {
	a1 = a1Val*Parsec;
	a2 = a2Val*Parsec;
	period = periodVal*Year;
	totalMass = ;
	massRatio = a1/a2;
	
	rPeriTot = rPeriTotVal*Parsec;
	m1 = m1Val*1.0e6*SolarMass;
	m2 = m2Val*1.0e6*SolarMass;
	rS1 = 2.0*G*m1/pow(c, 2.0);
	rS2 = 2.0*G*m2/pow(c, 2.0);
	//totalMass = m1 + m2;
	reducedMass = m1*m2/(m1 + m2);
	eccentricity = eccentricityVal;
	eccentricityFactor = sqrt((1.0 + eccentricity)/(1.0 - eccentricity));
	a1 = (m2*rPeriTot)/(totalMass*(1.0 - eccentricity));
	a2 = (m1*rPeriTot)/(totalMass*(1.0 - eccentricity));
	rPeri1 = a1*(1.0 - eccentricity);
	rPeri2 = a2*(1.0 - eccentricity);
	rApo1 = a1*(1.0 + eccentricity);
	rApo2 = a2*(1.0 + eccentricity);
	omega1 = d2r(omegaVal);
	omega2 = omega1 + pi;
	inclination = d2r(inclinationVal);
	tau = tauVal*Day;
	alpha1 = alpha1Val;
	alpha2 = alpha2Val;
	//period = twoPi*sqrt(pow(a1 + a2, 3.0)/(G*totalMass));
	epoch = 0.0;
	M = twoPi*(epoch - tau)/period;
	nlopt::opt opt(nlopt::LN_COBYLA, 1);
	KeplersEqnData Data;
	Data.eccentricity = eccentricity;
	Data.M = M;
	KeplersEqnData *ptr2Data = &Data;
	opt.set_min_objective(KeplerEqn, ptr2Data);
	opt.set_xtol_rel(1e-16);
	std::vector<double> x(1);
	x[0] = 1.0;
	double minf;
	nlopt::result result = opt.optimize(x, minf);
	E = x[0];
	r1 = a1*(1.0 - eccentricity*cos(E)); // current distance of m1 from COM
	r2 = (m1*r1)/m2; // current distance of m2 from COM
	nu = 2.0*atan(eccentricityFactor*tan(E/2.0)); // current true anomoly of m1
	if (nu < 0.0) {
		nu += 2.0*pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/c;
	beta2 = sqrt(((G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/c;
	_radialBetaFactor1 = (((twoPi/period)*a1)/(sqrt(1.0 - pow(eccentricity, 2.0))))*sin(inclination);
	_radialBetaFactor2 = (((twoPi/period)*a2)/(sqrt(1.0 - pow(eccentricity, 2.0))))*sin(inclination);
	radialBeta1 = (_radialBetaFactor1*(cos(nu + omega1) + eccentricity*cos(omega1)))/c;
	radialBeta2 = (_radialBetaFactor1*(cos(nu + omega2) + eccentricity*cos(omega2)))/c;
	dF1 = (sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1);
	dF2 = (sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2);
	bF1 = pow(dF1, 3.0 - alpha1);
	bF2 = pow(dF2, 3.0 - alpha2);
	totalFlux = 1.0;
	//fracBeamedFlux = 0.0;
	}

void binarySMBH::setBinarySMBH(double *Theta) {
	rPeriTot = Parsec*Theta[0];
	m1 = SolarMass*1.0e6*Theta[1];
	m2 = SolarMass*1.0e6*Theta[2];
	rS1 = (2.0*G*m1)/(pow(c, 2.0));
	rS2 = (2.0*G*m2)/(pow(c, 2.0));
	totalMass = m1 + m2;
	massRatio = m2/m1;
	reducedMass = m1*m2/(m1 + m2);
	eccentricity = Theta[3];
	eccentricityFactor = sqrt((1.0 + eccentricity)/(1.0 - eccentricity));
	a1 = (m2*rPeriTot)/(totalMass*(1.0 - eccentricity));
	a2 = (m1*rPeriTot)/(totalMass*(1.0 - eccentricity));
	rPeri1 = a1*(1.0 - eccentricity);
	rPeri2 = a2*(1.0 - eccentricity);
	rApo1 = a1*(1.0 + eccentricity);
	rApo2 = a2*(1.0 + eccentricity);
	omega1 = d2r(Theta[4]);
	omega2 = omega1 + pi;
	inclination = d2r(Theta[5]);
	tau = Theta[6]*Day;
	alpha1 = -0.44;
	alpha2 = -0.44;
	period = twoPi*sqrt(pow(a1 + a2, 3.0)/(G*totalMass));
	epoch = 0.0;
	M = twoPi*(epoch - tau)/period;
	nlopt::opt opt(nlopt::LN_COBYLA, 1);
	KeplersEqnData Data;
	Data.eccentricity = eccentricity;
	Data.M = M;
	KeplersEqnData *ptr2Data = &Data;
	opt.set_min_objective(KeplerEqn, ptr2Data);
	opt.set_xtol_rel(1e-16);
	std::vector<double> x(1);
	x[0] = 1.0;
	double minf;
	nlopt::result result = opt.optimize(x, minf);
	E = x[0];
	r1 = a1*(1.0 - eccentricity*cos(E)); // current distance of m1 from COM
	r2 = (m1*r1)/m2; // current distance of m2 from COM
	nu = 2.0*atan(eccentricityFactor*tan(E/2.0)); // current true anomoly of m1
	if (nu < 0.0) {
		nu += 2.0*pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/c;
	beta2 = sqrt(((G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/c;
	_radialBetaFactor1 = (((twoPi/period)*a1)/(sqrt(1.0 - pow(eccentricity, 2.0))))*sin(inclination);
	_radialBetaFactor2 = (((twoPi/period)*a2)/(sqrt(1.0 - pow(eccentricity, 2.0))))*sin(inclination);
	radialBeta1 = (_radialBetaFactor1*(cos(nu + omega1) + eccentricity*cos(omega1)))/c;
	radialBeta2 = (_radialBetaFactor1*(cos(nu + omega2) + eccentricity*cos(omega2)))/c;
	dF1 = (sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1);
	dF2 = (sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2);
	bF1 = pow(dF1, 3.0 - alpha1);
	bF2 = pow(dF2, 3.0 - alpha2);
	totalFlux = Theta[7];
	//fracBeamedFlux = Theta[8];
	}

void binarySMBH::operator()() {
	M = twoPi*(epoch - tau)/period;
	nlopt::opt opt(nlopt::LN_COBYLA, 1);
	KeplersEqnData Data;
	Data.eccentricity = eccentricity;
	Data.M = M;
	KeplersEqnData *ptr2Data = &Data;
	opt.set_min_objective(KeplerEqn, ptr2Data);
	opt.set_xtol_rel(1e-16);
	std::vector<double> x(1);
	x[0] = 1.0;
	double minf;
	nlopt::result result = opt.optimize(x, minf);
	E = x[0];
	r1 = a1*(1.0 - eccentricity*cos(E)); // current distance of m1 from COM
	r2 = (m1*r1)/m2; // current distance of m2 from COM
	nu = 2.0*atan(eccentricityFactor*tan(E/2.0)); // current true anomoly of m1
	if (nu < 0.0) {
		nu += 2.0*pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/c;
	beta2 = sqrt(((G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/c;
	radialBeta1 = (_radialBetaFactor1*(cos(nu + omega1) + eccentricity*cos(omega1)))/c;
	radialBeta2 = (_radialBetaFactor1*(cos(nu + omega2) + eccentricity*cos(omega2)))/c;
	dF1 = (sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1);
	dF2 = (sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2);
	bF1 = pow(dF1, 3.0 - alpha1);
	bF2 = pow(dF2, 3.0 - alpha2);
	}

int binarySMBH::checkBinarySMBHParams(double *ThetaIn) {
	int retVal = 1;

	double m1Val = 1.0e-6*SolarMass*ThetaIn[1];
	double m2Val = 1.0e-6*SolarMass*ThetaIn[2];
	if (m1Val < 1.0e-2*1.0e-6*SolarMass) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("m1: %4.3e\n",m1Val);
			printf("m1LLim: %4.3e\n",1.0e-1*1.0e-6*SolarMass);
		#endif
		}
	if (m2Val < 1.0e-2*1.0e-6*SolarMass) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("m2: %4.3e\n",m2Val);
			printf("m2LLim: %4.3e\n",1.0e-1*1.0e-6*SolarMass);
		#endif
		}
	if (m1Val > 1.0e4*1.0e-6*SolarMass) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("m1: %4.3e\n",m1Val);
			printf("m1ULim: %4.3e\n",1.0e3*1.0e-6*SolarMass);
		#endif
		}
	if (m2Val > m1Val) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("m2: %4.3e\n",m2Val);
			printf("m2ULim: %4.3e\n",m1Val);
		#endif
		}

	double rS1Val = (2.0*G*m1Val)/(pow(c, 2.0));
	double rS2Val = (2.0*G*m2Val)/(pow(c, 2.0));
	double rPerVal = Parsec*ThetaIn[0];
	if (rPerVal < (10.0*(rS1Val + rS2Val))) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("rPer: %+4.3e\n",rPerVal);
			printf("rPerLLim: %+4.3e\n", 10.0*(rS1Val + rS2Val));
		#endif
		}
	if (rPerVal > 10.0*Parsec) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("rPer: %4.3e\n",rPerVal);
			printf("rPerULim: %4.3e\n",1.0*Parsec);
		#endif
		}

	double eccentricityVal = ThetaIn[3];
	if (eccentricityVal < 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("eccentricity: %4.3e\n",eccentricityVal);
			printf("eccentricityLLim: %4.3e\n",0.0);
		#endif
		}
	if (eccentricityVal >= 1.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("eccentricity: %4.3e\n",eccentricityVal);
			printf("eccentricityULim: %4.3e\n",1.0);
		#endif
		}

	double omega1Val = d2r(ThetaIn[4]);
	if (omega1Val < 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("omega1: %4.3e\n",omega1Val);
			printf("omega1LLim: %4.3e\n",0.0);
		#endif
		}
	if (omega1Val >= twoPi) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("omega1: %4.3e\n",omega1Val);
			printf("omega1ULim: %4.3e\n",twoPi);
		#endif
		}

	double inclinationVal = d2r(ThetaIn[5]);
	if (inclinationVal < 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("inclination: %4.3e\n",inclinationVal);
			printf("inclinationLLim: %4.3e\n",0.0);
		#endif
		}
	if (inclinationVal >= pi) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("inclination: %4.3e\n",inclinationVal);
			printf("inclinationULim: %4.3e\n",pi);
		#endif
		}

	double periodVal = twoPi*sqrt(pow(rPerVal/(1.0 + eccentricityVal), 3.0)/(G*(m1Val + m2Val)));
	double tauVal = ThetaIn[6]*Day;
	if (tauVal < 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("tau: %4.3e\n",tauVal);
			printf("tauLLim: %4.3e\n",0.0);
		#endif
		}
	if (tauVal > periodVal) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("tau: %4.3e\n",tauVal);
			printf("tauULim: %4.3e\n",periodVal);
		#endif
		}

	double totalFluxVal = ThetaIn[7];
	if (totalFluxVal < 0.0) {
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("totalFlux: %4.3e\n",totalFluxVal);
			printf("totalFluxLLim: %4.3e\n",0.0);
		#endif
		retVal = 0;
		}

	/*double fracBeamedFluxVal = ThetaIn[8];
	if (fracBeamedFluxVal <= 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("fracBeamedFlux: %4.3e\n",fracBeamedFluxVal);
			printf("fracBeamedFluxLLim: %4.3e\n",0.0);
		#endif
		}
	if (fracBeamedFluxVal > 1.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKBINARYSMBHPARAMS
			printf("fracBeamedFlux: %4.3e\n",fracBeamedFluxVal);
			printf("fracBeamedFluxULim: %4.3e\n",1.0);
		#endif
		}*/

	return retVal;
	}

double binarySMBH::getEpoch() {return epoch/Day;}

void binarySMBH::setEpoch(double epochIn) {
	epoch = epochIn*Day;
	(*this)();
	}

double binarySMBH::getPeriod() {return period/Day;}

double binarySMBH::getA1() {return a1/Parsec;}

double binarySMBH::getA2() {return a2/Parsec;}

double binarySMBH::getM1() {return m1/(SolarMass*1.0e6);}

double binarySMBH::getM2() {return m2/(SolarMass*1.0e6);}

double binarySMBH::getRPeri1() {return rPeri1/Parsec;}

double binarySMBH::getRPeri2() {return rPeri2/Parsec;}

double binarySMBH::getRPeriTot() {return rPeriTot/Parsec;}

double binarySMBH::getRApoTot() {return rApoTot/Parsec;}

double binarySMBH::getRApo1() {return rApo1/Parsec;}

double binarySMBH::getRApo2() {return rApo2/Parsec;}

double binarySMBH::getRS1() {return rS1/Parsec;}

double binarySMBH::getRS2() {return rS2/Parsec;}

double binarySMBH::getEccentricity() {return eccentricity;}

double binarySMBH::getOmega1() {return r2d(omega1);}

double binarySMBH::getOmega2() {return r2d(omega2);}

double binarySMBH::getInclination() {return r2d(inclination);}

double binarySMBH::getTau() {return tau/Day;}

double binarySMBH::getMeanAnomoly() {return r2d(M);}

double binarySMBH::getEccentricAnomoly() {return r2d(E);}

double binarySMBH::getTrueAnomoly() {return r2d(nu);}

double binarySMBH::getR1() {return r1/Parsec;}

double binarySMBH::getR2() {return r2/Parsec;}

double binarySMBH::getTheta1() {return r2d(theta1);}

double binarySMBH::getTheta2() {return r2d(theta2);}

double binarySMBH::getBeta1() {return beta1;}

double binarySMBH::getBeta2() {return beta2;}

double binarySMBH::getRadialBeta1() {return radialBeta1;}

double binarySMBH::getRadialBeta2() {return radialBeta2;}

double binarySMBH::getDopplerFactor1() {return dF1;}

double binarySMBH::getDopplerFactor2() {return dF2;}

double binarySMBH::getBeamingFactor1() {return bF1;}

double binarySMBH::getBeamingFactor2() {return bF2;}

double binarySMBH::aH(double sigmaStars) {
	double aHVal = (G*reducedMass)/(4.0*pow((sigmaStars*kms2ms), 2.0));
	return aHVal/Parsec;
	}

double binarySMBH::aGW(double sigmaStars, double rhoStars, double H) {
	double aGWVal = pow((64.0*pow(G*reducedMass, 2.0)*totalMass*(kms2ms*sigmaStars))/(5.0*H*pow(c, 5.0)*(SolarMassPerCubicParsec*rhoStars)), 0.2);
	return aGWVal/Parsec;
	}

double binarySMBH::durationInHardState(double sigmaStars, double rhoStars, double H) {
	double durationInHardStateVal = ((sigmaStars*kms2ms)/(H*G*(SolarMassPerCubicParsec*rhoStars)*aGW(sigmaStars, rhoStars, H)));
	return durationInHardStateVal/Day;
	}

double binarySMBH::ejectedMass(double sigmaStars, double rhoStars, double H) {
	double ejectedMassVal = totalMass*log(aH(sigmaStars)/aGW(sigmaStars, rhoStars, H));
	return ejectedMassVal/(SolarMass*1.0e6);
	}

void binarySMBH::print() {
	cout.precision(5);
	cout << scientific << "               rPeri1 + rPeri2: " << rPeriTot/Parsec << " (pc)" << endl;
	cout << scientific << "                            a1: " << a1/Parsec << " (pc)" << endl;
	cout << scientific << "                            a2: " << a2/Parsec << " (pc)" << endl;
	cout << scientific << "                            m1: " << m1/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                            m2: " << m2/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                           rS1: " << rS1/Parsec << " (pc)" << endl;
	cout << scientific << "                           rS2: " << rS2/Parsec << " (pc)" << endl;
	cout << scientific << "                    Total Mass: " << totalMass/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                    Mass Ratio: " << massRatio << endl;
	cout << scientific << "                  Reduced Mass: " << reducedMass/(1.0e6*SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                        Period: " << period/Day << " (day) == " << period/Year  << " (year)" << endl;
	cout << scientific << "                  Eccentricity: " << eccentricity << endl;
	cout << scientific << "Argument of periapsis (mass 1): " << r2d(omega1) << " (degree)" << endl;
	cout << scientific << "                   Inclination: " << r2d(inclination) << " (degree)" << endl;
	cout << scientific << "            Time of Periastron: " << tau/Day << " (day)" << endl;
	cout << scientific << "                    Total Flux: " << totalFlux << endl;
	//cout << scientific << "          Beamed Flux Fraction: " << fracBeamedFlux << endl;
	}

void binarySMBH::simulateSystem(LnLikeData *ptr2Data) {
	LnLikeData Data = *ptr2Data;

	int numCadences = Data.numCadences;
	double *t = Data.t;
	double *x = Data.x;

	for (int i = 0; i < numCadences; ++i) {
		setEpoch(t[i]);
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
		//x[i] = totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2());
		x[i] = totalFlux*getBeamingFactor2();
		#ifdef DEBUG_SIMULATESYSTEM
			printf("totalFlux*getBeamingFactor2(): %e\n",totalFlux*getBeamingFactor2());
			//printf("totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()): %e\n",totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()));
		#endif
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
		#ifdef DEBUG_OBSERVENOISE
			printf("noiseLvl: %e\n", noiseLvl);
		#endif
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, noiseStream, 1, &noiseRand[i], 0.0, noiseLvl);
		#ifdef DEBUG_OBSERVENOISE
			printf("noiseRand[%d]: %e\n", i, noiseRand[i]);
			printf("x[%d]: %e\n",i, x[i]);
		#endif
		y[i] = x[i] + noiseRand[i];
		#ifdef DEBUG_OBSERVENOISE
			printf("y[%d]: %e\n",i, y[i]);
			printf("y[%d] - x[%d]: %+9.8e\n",i, i, y[i] - x[i]);
		#endif
		yerr[i] = noiseLvl;
		}
	}

double binarySMBH::computeLnPrior(LnLikeData *ptr2Data) {
	LnLikeData Data = *ptr2Data;

	int numCadences = Data.numCadences;
	double dt = Data.dt*Day;
	double currentLnPrior = Data.currentLnPrior;
	double *t = Data.t;
	double T = (t[numCadences-1] - t[0])*Day;
	double *y = Data.y;
	double *yerr = Data.yerr;
	double *mask = Data.mask;
	double lowestFlux = Data.lowestFlux;
	double highestFlux = Data.highestFlux;
	double maxDouble = numeric_limits<double>::max();

	#ifdef DEBUG_COMPUTELNPRIOR
	int threadNum = omp_get_thread_num();
	#endif

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	double LnPrior = 0.0, timescale = 0.0, timescaleOsc = 0.0;

	if (totalFlux < lowestFlux) {
		LnPrior = -infiniteVal; // The total flux cannot be smaller than the smallest flux in the LC
		}
	if (totalFlux > highestFlux) {
		LnPrior = -infiniteVal; // The total flux cannot be bigger than the biggest flux in the LC
		}

	if (period < 2.0*dt) {
		LnPrior = -infiniteVal; // Cut all all configurations where the inferred period is too short!
		}

	if (period > 10.0*T) {
		LnPrior = -infiniteVal; // Cut all all configurations where the inferred period is too short!
		}

	Data.currentLnPrior = LnPrior;

	return LnPrior;
	}

double binarySMBH::computeLnLikelihood(LnLikeData *ptr2Data) {
	LnLikeData Data = *ptr2Data;

	int numCadences = Data.numCadences;
	double dt = Day*Data.dt;
	double *t = Data.t;
	double *x = Data.x;
	double *y = Data.y;
	double *yerr = Data.yerr;
	double *mask = Data.mask;
	double maxDouble = numeric_limits<double>::max();

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	double LnLikelihood = 0.0, Contrib = 0.0, ptCounter = 0.0;

	for (int i = 0; i < numCadences; ++i) {
		setEpoch(t[i]);
		//Contrib = mask[i]*(-1.0*log2(yerr[i])/log2OfE -0.5*pow(((y[i] - totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()))/yerr[i]), 2.0));
		Contrib = mask[i]*(-1.0*log2(yerr[i])/log2OfE -0.5*pow(((y[i] - totalFlux*getBeamingFactor2())/yerr[i]), 2.0));
		#ifdef DEBUG_COMPUTELNLIKELIHOOD
			printf("y[i]: %e\n", y[i]);
			printf("yerr[i]: %e\n", yerr[i]);
			//printf("totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()): %e\n", totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()));
			//printf("x[i] - totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()): %e\n", x[i] - totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()));
			//printf("y[i] - totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()): %e\n", y[i] - totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()));
			//printf("-0.5*pow(((y[i] - totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()))/yerr[i]), 2.0): %e\n", -0.5*pow(((y[i] - totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()))/yerr[i]), 2.0));
			printf("totalFlux*getBeamingFactor2(): %e\n", totalFlux*getBeamingFactor2());
			printf("x[i] - totalFlux*getBeamingFactor2(): %e\n", x[i] - totalFlux*getBeamingFactor2());
			printf("y[i] - totalFlux*getBeamingFactor2(): %e\n", y[i] - totalFlux*getBeamingFactor2());
			printf("-0.5*pow(((y[i] - totalFlux*getBeamingFactor2())/yerr[i]), 2.0): %e\n", -0.5*pow(((y[i] - totalFlux*getBeamingFactor2())/yerr[i]), 2.0));
			printf("-1.0*log2(yerr[i])/log2OfE): %e\n",-1.0*log2(yerr[i])/log2OfE);
			printf("Contrib: %e\n",Contrib);
		#endif
		LnLikelihood = LnLikelihood + Contrib;
		#ifdef DEBUG_COMPUTELNLIKELIHOOD
			printf("LnLikelihood: %e\n",LnLikelihood);
		#endif
		ptCounter += mask[i];
		}
	#ifdef DEBUG_COMPUTELNLIKELIHOOD
		printf("-0.5*ptCounter*log2Pi: %e\n",-0.5*ptCounter*log2Pi);
	#endif
	LnLikelihood += -0.5*ptCounter*log2Pi;
	#ifdef DEBUG_COMPUTELNLIKELIHOOD
		printf("LnLikelihood: %e\n",LnLikelihood);
	#endif

	Data.cadenceNum = numCadences - 1;
	Data.currentLnLikelihood = LnLikelihood;
	return LnLikelihood;
	}