#ifdef __INTEL_COMPILER
    #include <mathimf.h>
    #if defined __APPLE__ && defined __MACH__
        #include <malloc/malloc.h>
    #else
        #include <malloc.h>
    #endif
#else
    #include <math.h>
    #include <mm_malloc.h>
#endif
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <nlopt.hpp>
#include <limits>
#include <vector>
#include <iostream>
#include <stdio.h>
#include "Constants.hpp"
#include "MBHB.hpp"

//#define DEBUG_KEPLEREQN
//#define DEBUG_SIMULATESYSTEM
//#define DEBUG_CHECKMBHBPARAMS
//#define DEBUG_CALCLNPOSTERIOR
//#define DEBUG_COMPUTELNPRIOR

//#if defined(DEBUG_KEPLEREQN) || defined(DEBUG_SIMULATESYSTEM) || defined(DEBUG_CHECKMBHBPARAMS) || defined(DEBUG_CALCLNPOSTERIOR)
//	#include <stdio.h>
//#endif

int lenThetaAlso = 8;

using namespace std;

double kali::calcLnPrior(const vector<double> &x, vector<double>& grad, void *p2Args) {
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
	MBHB *Systems = Args.Systems;
	double LnPrior = 0.0;

	if (Systems[threadNum].checkMBHBParams(const_cast<double*>(&x[0])) == 1) {
		LnPrior = 0.0;
		} else {
		LnPrior = -infiniteVal;
		}
	return LnPrior;
	}

double kali::calcLnPrior(double *walkerPos, void *func_args) {
	/*! Used for computing good regions */
	int threadNum = omp_get_thread_num();
	LnLikeArgs *ptr2Args = reinterpret_cast<LnLikeArgs*>(func_args);
	LnLikeArgs Args = *ptr2Args;
	MBHB* Systems = Args.Systems;
	double LnPrior = 0.0;
	if (Systems[threadNum].checkMBHBParams(walkerPos) == 1) {
		LnPrior = 0.0;
		} else {
		LnPrior = -infiniteVal;
		}
	return LnPrior;
	}

double kali::calcLnPosterior(const vector<double> &x, vector<double>& grad, void *p2Args) {
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
	MBHB *Systems = Args.Systems;
	double LnPosterior = 0.0, old_dt = 0.0;
	if (Systems[threadNum].checkMBHBParams(const_cast<double*>(&x[0])) == 1) {
		Systems[threadNum].setMBHB(const_cast<double*>(&x[0]));
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

double kali::calcLnPosterior(double *walkerPos, void *func_args, double &LnPrior, double &LnLikelihood) {
	int threadNum = omp_get_thread_num();
	LnLikeArgs *ptr2Args = reinterpret_cast<LnLikeArgs*>(func_args);
	LnLikeArgs Args = *ptr2Args;
	LnLikeData *Data = Args.Data;
	MBHB *Systems = Args.Systems;
	LnLikeData *ptr2Data = Data;
	double LnPosterior = 0.0, old_dt = 0.0;
	if (Systems[threadNum].checkMBHBParams(walkerPos) == 1) {
		Systems[threadNum].setMBHB(walkerPos);
        LnPrior = Systems[threadNum].computeLnPrior(ptr2Data);

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

        LnLikelihood = Systems[threadNum].computeLnLikelihood(ptr2Data);
		LnPosterior = LnLikelihood + LnPrior;

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
        LnPrior = -kali::infiniteVal;
        LnLikelihood = -kali::infiniteVal;;
    	LnPosterior = -kali::infiniteVal;
		}
	return LnPosterior;
	}

double kali::d2r(double degreeVal) {
	return degreeVal*(pi/180.0);
	}

double kali::r2d(double radianVal) {
	return radianVal*(180.0/pi);
	}

double kali::KeplerEqn(const vector<double> &x, vector<double> &grad, void *p2Data) {
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

kali::MBHB::MBHB() {
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
	rPeribothron1 = 0.0;
	rPeribothron2 = 0.0;
	rApobothron1 = 0.0;
	rApobothron2 = 0.0;
	rPeribothronTot = 0.0;
	rApobothronTot = 0.0;
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

kali::MBHB::MBHB(double a1Val, double a2Val, double periodVal, double eccentricityVal, double omegaVal, double inclinationVal, double tauVal, double alpha1Val, double alpha2Val) {
	a1 = a1Val*kali::Parsec;
	a2 = a2Val*kali::Parsec;
	period = periodVal*kali::Day;
	totalMass = (kali::fourPiSq*pow(a1 + a2, 3.0))/(kali::G*pow(period, 2.0));
	massRatio = a1/a2;
	m1 = totalMass*(1.0/(1.0 + massRatio));
	m2 = totalMass*(massRatio/(1.0 + massRatio));
	rS1 = 2.0*kali::G*m1/pow(kali::c, 2.0);
	rS2 = 2.0*kali::G*m2/pow(kali::c, 2.0);
	reducedMass = m1*m2/(m1 + m2);
	eccentricity = eccentricityVal;
	eccentricityFactor = sqrt((1.0 + eccentricity)/(1.0 - eccentricity));
	rPeribothron1 = a1*(1.0 - eccentricity);
	rPeribothron2 = a2*(1.0 - eccentricity);
	rPeribothronTot = rPeribothron1 + rPeribothron2;
	rApobothron1 = a1*(1.0 + eccentricity);
	rApobothron2 = a2*(1.0 + eccentricity);
	rApobothronTot = rApobothron1 + rApobothron2;
	omega1 = d2r(omegaVal);
	omega2 = omega1 + kali::pi;
	inclination = d2r(inclinationVal);
	tau = tauVal*kali::Day;
	alpha1 = alpha1Val;
	alpha2 = alpha2Val;
	epoch = 0.0;
	M = kali::twoPi*(epoch - tau)/period;
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
		nu += 2.0*kali::pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((kali::G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/kali::c;
	beta2 = sqrt(((kali::G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/kali::c;
	_radialBetaFactor1 = (((kali::twoPi/period)*a1)/(sqrt(1.0 - pow(eccentricity, 2.0))))*sin(inclination);
	_radialBetaFactor2 = (((kali::twoPi/period)*a2)/(sqrt(1.0 - pow(eccentricity, 2.0))))*sin(inclination);
	radialBeta1 = (_radialBetaFactor1*(cos(nu + omega1) + eccentricity*cos(omega1)))/kali::c;
	radialBeta2 = (_radialBetaFactor1*(cos(nu + omega2) + eccentricity*cos(omega2)))/kali::c;
	dF1 = (sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1);
	dF2 = (sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2);
	bF1 = pow(dF1, 3.0 - alpha1);
	bF2 = pow(dF2, 3.0 - alpha2);
	totalFlux = 1.0;
	//fracBeamedFlux = 0.0;
	}

void kali::MBHB::setMBHB(double *Theta) {
	a1 = Theta[0]*kali::Parsec;
	a2 = Theta[1]*kali::Parsec;
	period = Theta[2]*kali::Day;
	totalMass = (kali::fourPiSq*pow(a1 + a2, 3.0))/(kali::G*pow(period, 2.0));
	massRatio = a1/a2;
	m1 = totalMass*(1.0/(1.0 + massRatio));
	m2 = totalMass*(massRatio/(1.0 + massRatio));
	rS1 = (2.0*kali::G*m1)/(pow(kali::c, 2.0));
	rS2 = (2.0*kali::G*m2)/(pow(kali::c, 2.0));
	reducedMass = m1*m2/(m1 + m2);
	eccentricity = Theta[3];
	eccentricityFactor = sqrt((1.0 + eccentricity)/(1.0 - eccentricity));
	rPeribothron1 = a1*(1.0 - eccentricity);
	rPeribothron2 = a2*(1.0 - eccentricity);
	rPeribothronTot = rPeribothron1 + rPeribothron2;
	rApobothron1 = a1*(1.0 + eccentricity);
	rApobothron2 = a2*(1.0 + eccentricity);
	rApobothronTot = rApobothron1 + rApobothron2;
	omega1 = d2r(Theta[4]);
	omega2 = omega1 + kali::pi;
	inclination = d2r(Theta[5]);
	tau = Theta[6]*kali::Day;
	alpha1 = -0.44;
	alpha2 = -0.44;
	period = kali::twoPi*sqrt(pow(a1 + a2, 3.0)/(kali::G*totalMass));
	epoch = 0.0;
	M = kali::twoPi*(epoch - tau)/period;
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
		nu += 2.0*kali::pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((kali::G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/kali::c;
	beta2 = sqrt(((kali::G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/kali::c;
	_radialBetaFactor1 = (((kali::twoPi/period)*a1)/(sqrt(1.0 - pow(eccentricity, 2.0))))*sin(inclination);
	_radialBetaFactor2 = (((kali::twoPi/period)*a2)/(sqrt(1.0 - pow(eccentricity, 2.0))))*sin(inclination);
	radialBeta1 = (_radialBetaFactor1*(cos(nu + omega1) + eccentricity*cos(omega1)))/kali::c;
	radialBeta2 = (_radialBetaFactor1*(cos(nu + omega2) + eccentricity*cos(omega2)))/kali::c;
	dF1 = (sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1);
	dF2 = (sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2);
	bF1 = pow(dF1, 3.0 - alpha1);
	bF2 = pow(dF2, 3.0 - alpha2);
	totalFlux = Theta[7];
	//fracBeamedFlux = Theta[8];
	}

void kali::MBHB::operator()() {
	M = kali::twoPi*(epoch - tau)/period;
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
		nu += 2.0*kali::pi;
		}
	theta1 = nu + omega1;
	theta2 = nu + omega2;
	beta1 = sqrt(((kali::G*pow(m2, 2.0))/totalMass)*((2.0/r1) - (1.0/a1)))/kali::c;
	beta2 = sqrt(((kali::G*pow(m1, 2.0))/totalMass)*((2.0/r2) - (1.0/a2)))/kali::c;
	radialBeta1 = (_radialBetaFactor1*(cos(nu + omega1) + eccentricity*cos(omega1)))/kali::c;
	radialBeta2 = (_radialBetaFactor1*(cos(nu + omega2) + eccentricity*cos(omega2)))/kali::c;
	dF1 = (sqrt(1.0 - pow(beta1, 2.0)))/(1.0 - radialBeta1);
	dF2 = (sqrt(1.0 - pow(beta2, 2.0)))/(1.0 - radialBeta2);
	bF1 = pow(dF1, 3.0 - alpha1);
	bF2 = pow(dF2, 3.0 - alpha2);
	}

int kali::MBHB::checkMBHBParams(double *ThetaIn) {
	int retVal = 1;
	double a1Val = kali::Parsec*ThetaIn[0];
	double a2Val = kali::Parsec*ThetaIn[1];

	if (a2Val < a1Val) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("a2: %4.3e\n",a2Val);
			printf("a2LLim: %4.3e\n",a1Val);
		#endif
		}

	double periodVal = kali::Day*ThetaIn[2];

    if (periodVal <= 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBCARMAPARAMS
			printf("period: %4.3e\n",periodVal);
			printf("periodLLim: %4.3e\n",0.0);
		#endif
		}

	double totalMassVal = (kali::fourPiSq*pow(a1Val + a2Val, 3.0))/(kali::G*pow(periodVal, 2.0));
	double massRatioVal = a1Val/a2Val;
	double m1Val = totalMassVal*(1.0/(1.0 + massRatioVal));
	double m2Val = totalMassVal*(massRatioVal/(1.0 + massRatioVal));

    if (m1Val < 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("m1: %4.3e\n",m1Val);
			printf("m1LLim: %4.3e\n",0.0);
		#endif
		}

    if (m2Val < 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("m2: %4.3e\n",m2Val);
			printf("m2LLim: %4.3e\n",0.0);
		#endif
		}

	double rS1Val = (2.0*kali::G*m1Val)/(pow(kali::c, 2.0));
	double rS2Val = (2.0*kali::G*m2Val)/(pow(kali::c, 2.0));

	double eccentricityVal = ThetaIn[3];
	if (eccentricityVal < 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("eccentricity: %4.3e\n",eccentricityVal);
			printf("eccentricityLLim: %4.3e\n",0.0);
		#endif
		}
	if (eccentricityVal >= 1.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("eccentricity: %4.3e\n",eccentricityVal);
			printf("eccentricityULim: %4.3e\n",1.0);
		#endif
		}

	double rPeribothronTotVal = a1Val*(1.0 - eccentricityVal) + a2Val*(1.0 - eccentricityVal);
	if (rPeribothronTotVal < (10.0*(rS1Val + rS2Val))) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("rPeribothronTot: %+4.3e\n",rPeribothronTotVal);
			printf("rPeribothronTotLLim: %+4.3e\n", 10.0*(rS1Val + rS2Val));
		#endif
		}

	double omega1Val = d2r(ThetaIn[4]);
	if (omega1Val < 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("omega1: %4.3e\n",omega1Val);
			printf("omega1LLim: %4.3e\n",0.0);
		#endif
		}
	if (omega1Val >= kali::twoPi) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("omega1: %4.3e\n",omega1Val);
			printf("omega1ULim: %4.3e\n",kali::twoPi);
		#endif
		}

	double inclinationVal = d2r(ThetaIn[5]);
	if (inclinationVal < 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("inclination: %4.3e\n",inclinationVal);
			printf("inclinationLLim: %4.3e\n",0.0);
		#endif
		}
	if (inclinationVal > kali::halfPi) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("inclination: %4.3e\n",inclinationVal);
			printf("inclinationULim: %4.3e\n",kali::halfPi);
		#endif
		}

	double tauVal = ThetaIn[6]*kali::Day;

	double totalFluxVal = ThetaIn[7];
	if (totalFluxVal < 0.0) {
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("totalFlux: %4.3e\n",totalFluxVal);
			printf("totalFluxLLim: %4.3e\n",0.0);
		#endif
		retVal = 0;
		}

	/*double fracBeamedFluxVal = ThetaIn[8];
	if (fracBeamedFluxVal <= 0.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("fracBeamedFlux: %4.3e\n",fracBeamedFluxVal);
			printf("fracBeamedFluxLLim: %4.3e\n",0.0);
		#endif
		}
	if (fracBeamedFluxVal > 1.0) {
		retVal = 0;
		#ifdef DEBUG_CHECKMBHBPARAMS
			printf("fracBeamedFlux: %4.3e\n",fracBeamedFluxVal);
			printf("fracBeamedFluxULim: %4.3e\n",1.0);
		#endif
		}*/

	return retVal;
	}

double kali::MBHB::getEpoch() {return epoch/kali::Day;}

void kali::MBHB::setEpoch(double epochIn) {
	epoch = epochIn*kali::Day;
	(*this)();
	}

double kali::MBHB::getPeriod() {return period/kali::Day;}

double kali::MBHB::getA1() {return a1/kali::Parsec;}

double kali::MBHB::getA2() {return a2/kali::Parsec;}

double kali::MBHB::getM1() {return m1/(kali::SolarMass*1.0e6);}

double kali::MBHB::getM2() {return m2/(kali::SolarMass*1.0e6);}

double kali::MBHB::getM12() {return totalMass/(kali::SolarMass*1.0e6);}

double kali::MBHB::getM2OverM1() {return massRatio;}

double kali::MBHB::getRPeribothron1() {return rPeribothron1/kali::Parsec;}

double kali::MBHB::getRPeribothron2() {return rPeribothron2/kali::Parsec;}

double kali::MBHB::getRPeribothronTot() {return rPeribothronTot/kali::Parsec;}

double kali::MBHB::getRApobothron1() {return rApobothron1/kali::Parsec;}

double kali::MBHB::getRApobothron2() {return rApobothron2/kali::Parsec;}

double kali::MBHB::getRApobothronTot() {return rApobothronTot/kali::Parsec;}

double kali::MBHB::getRS1() {return rS1/kali::Parsec;}

double kali::MBHB::getRS2() {return rS2/kali::Parsec;}

double kali::MBHB::getEccentricity() {return eccentricity;}

double kali::MBHB::getOmega1() {return r2d(omega1);}

double kali::MBHB::getOmega2() {return r2d(omega2);}

double kali::MBHB::getInclination() {return r2d(inclination);}

double kali::MBHB::getTau() {return tau/kali::Day;}

double kali::MBHB::getMeanAnomoly() {return r2d(M);}

double kali::MBHB::getEccentricAnomoly() {return r2d(E);}

double kali::MBHB::getTrueAnomoly() {return r2d(nu);}

double kali::MBHB::getR1() {return r1/kali::Parsec;}

double kali::MBHB::getR2() {return r2/kali::Parsec;}

double kali::MBHB::getTheta1() {return r2d(theta1);}

double kali::MBHB::getTheta2() {return r2d(theta2);}

double kali::MBHB::getBeta1() {return beta1;}

double kali::MBHB::getBeta2() {return beta2;}

double kali::MBHB::getRadialBeta1() {return radialBeta1;}

double kali::MBHB::getRadialBeta2() {return radialBeta2;}

double kali::MBHB::getDopplerFactor1() {return dF1;}

double kali::MBHB::getDopplerFactor2() {return dF2;}

double kali::MBHB::getBeamingFactor1() {return bF1;}

double kali::MBHB::getBeamingFactor2() {return bF2;}

double kali::MBHB::aH(double sigmaStars) {
	double aHVal = (kali::G*reducedMass)/(4.0*pow((sigmaStars*kali::kms2ms), 2.0));
	return aHVal/kali::Parsec;
	}

double kali::MBHB::aGW(double sigmaStars, double rhoStars, double H) {
	double aGWVal = pow((64.0*pow(kali::G*reducedMass, 2.0)*totalMass*(kali::kms2ms*sigmaStars))/(5.0*H*pow(kali::c, 5.0)*(kali::SolarMassPerCubicParsec*rhoStars)), 0.2);
	return aGWVal/kali::Parsec;
	}

double kali::MBHB::durationInHardState(double sigmaStars, double rhoStars, double H) {
	double durationInHardStateVal = ((sigmaStars*kali::kms2ms)/(H*kali::G*(kali::SolarMassPerCubicParsec*rhoStars)*aGW(sigmaStars, rhoStars, H)));
	return durationInHardStateVal/kali::Day;
	}

double kali::MBHB::ejectedMass(double sigmaStars, double rhoStars, double H) {
	double ejectedMassVal = totalMass*log(aH(sigmaStars)/aGW(sigmaStars, rhoStars, H));
	return ejectedMassVal/(kali::SolarMass*1.0e6);
	}

void kali::MBHB::print() {
	cout.precision(5);
	cout << scientific << "                            a1: " << a1/kali::Parsec << " (pc)" << endl;
	cout << scientific << "                            a2: " << a2/kali::Parsec << " (pc)" << endl;
	cout << scientific << "                        Period: " << period/kali::Day << " (day) == " << period/kali::Year  << " (year)" << endl;
	cout << scientific << "                            m1: " << m1/(1.0e6*kali::SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                            m2: " << m2/(1.0e6*kali::SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                    Total Mass: " << totalMass/(1.0e6*kali::SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                    Mass Ratio: " << massRatio << endl;
	cout << scientific << "                  Reduced Mass: " << reducedMass/(1.0e6*kali::SolarMass) << " (10^6 Solar Mass)" << endl;
	cout << scientific << "                           rS1: " << rS1/kali::Parsec << " (pc)" << endl;
	cout << scientific << "                           rS2: " << rS2/kali::Parsec << " (pc)" << endl;
	cout << scientific << "                  Eccentricity: " << eccentricity << endl;
	cout << scientific << "                  rPeribothron: " << rPeribothronTot/kali::Parsec << " (pc)" << endl;
	cout << scientific << "                   rApobothron: " << rApobothronTot/kali::Parsec << " (pc)" << endl;
	cout << scientific << "Argument of periapsis (mass 1): " << r2d(omega1) << " (degree)" << endl;
	cout << scientific << "                   Inclination: " << r2d(inclination) << " (degree)" << endl;
	cout << scientific << "            Time of Periastron: " << tau/kali::Day << " (day)" << endl;
	cout << scientific << "                    Total Flux: " << totalFlux << endl;
	//cout << scientific << "          Beamed Flux Fraction: " << fracBeamedFlux << endl;
	}

void kali::MBHB::simulateSystem(LnLikeData *ptr2Data) {
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

void kali::MBHB::observeNoise(LnLikeData *ptr2Data, unsigned int noiseSeed, double* noiseRand) {
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

double kali::MBHB::computeLnPrior(LnLikeData *ptr2Data) {
	LnLikeData Data = *ptr2Data;

	int numCadences = Data.numCadences;
	double dt = Data.dt*kali::Day;
	double currentLnPrior = Data.currentLnPrior;
	double *t = Data.t;
	double T = (t[numCadences-1] - t[0])*kali::Day;
	double *y = Data.y;
	double *yerr = Data.yerr;
	double *mask = Data.mask;
    double startT = Data.startT*kali::Day;
	double lowestFlux = Data.lowestFlux;
	double highestFlux = Data.highestFlux;
    double periodCenter = Data.periodCenter;
    double periodWidth = Data.periodWidth;
    double fluxCenter = Data.fluxCenter;
    double fluxWidth = Data.fluxWidth;
	double maxDouble = numeric_limits<double>::max();

	#ifdef DEBUG_COMPUTELNPRIOR
	int threadNum = omp_get_thread_num();
	#endif

	mkl_domain_set_num_threads(1, MKL_DOMAIN_ALL);
	double LnPrior = 0.0, timescale = 0.0, timescaleOsc = 0.0;

    if (m1 < 1.0e4*kali::SolarMass) {
		LnPrior = -infiniteVal; // Restrict m1 > 10^4 M_Sum
		}
	if (m1 > 1.0e10*kali::SolarMass) {
		LnPrior = -infiniteVal; // Restrict m1 < 10^10 M_Sun
		}

    #ifdef DEBUG_COMPUTELNPRIOR
        printf("computeLnPrior - threadNum: %d; LnPrior after m1: %+4.3e\n",threadNum,LnPrior);
    #endif

    if (m2 < 1.0e4*kali::SolarMass) {
		LnPrior = -infiniteVal; // Restrict m2 > 10^4 M_Sum
		}
	if (m2 > m1) {
		LnPrior = -infiniteVal; // Restrict m1 >= m2
		}

    #ifdef DEBUG_COMPUTELNPRIOR
        printf("computeLnPrior - threadNum: %d; LnPrior after m2: %+4.3e\n",threadNum,LnPrior);
    #endif

    if (tau < startT) {
		LnPrior = -infiniteVal; // Restrict tau to startT < tau < startT + period
		}
	if (tau > startT + period) {
		LnPrior = -infiniteVal; // Restrict tau to startT < tau < startT + period
		}

    #ifdef DEBUG_COMPUTELNPRIOR
        printf("computeLnPrior - threadNum: %d; LnPrior after tau: %+4.3e\n",threadNum,LnPrior);
    #endif

	if (totalFlux < lowestFlux) {
		LnPrior = -infiniteVal; // The total flux cannot be smaller than the smallest flux in the LC
		}
	if (totalFlux > highestFlux) {
		LnPrior = -infiniteVal; // The total flux cannot be bigger than the biggest flux in the LC
		}

    #ifdef DEBUG_COMPUTELNPRIOR
        printf("computeLnPrior - threadNum: %d; LnPrior after flux: %+4.3e\n",threadNum,LnPrior);
    #endif

	if (period < 2.0*dt) {
		LnPrior = -infiniteVal; // Cut all all configurations where the inferred period is too short!
		}

	if (period > 10.0*T) {
		LnPrior = -infiniteVal; // Cut all all configurations where the inferred period is too short!
		}

    #ifdef DEBUG_COMPUTELNPRIOR
        printf("computeLnPrior - threadNum: %d; LnPrior after period: %+4.3e\n",threadNum,LnPrior);
    #endif

    double periodPrior = -0.5*pow(((periodCenter*kali::Day) - period)/(periodWidth*kali::Day), 2.0); // Prior on period
    double totalFluxPrior = -0.5*pow((fluxCenter - totalFlux)/fluxWidth, 2.0); // Prior on totalFlux
    LnPrior += periodPrior + totalFluxPrior;

    #ifdef DEBUG_COMPUTELNPRIOR
    	printf("computeLnPrior - threadNum: %d; totalFlux: %+e\n", threadNum, totalFlux);
        printf("computeLnPrior - threadNum: %d; lowestFlux: %+e\n", threadNum, lowestFlux);
        printf("computeLnPrior - threadNum: %d; highestFlux: %+e\n", threadNum, highestFlux);
        printf("computeLnPrior - threadNum: %d; period: %+e\n", threadNum, period);
        printf("computeLnPrior - threadNum: %d; 2.0*dt: %+e\n", threadNum, 2.0*dt);
        printf("computeLnPrior - threadNum: %d; 10.0*T: %+e\n", threadNum, 10.0*T);
        printf("computeLnPrior - threadNum: %d; period: %+e\n", threadNum, period);
        printf("computeLnPrior - threadNum: %d; periodCenter: %+e\n", threadNum, periodCenter*kali::Day);
        printf("computeLnPrior - threadNum: %d; periodWidth: %+e\n", threadNum, periodWidth*kali::Day);
        //printf("computeLnPrior - threadNum: %d;                                                                -0.5 ln 2pi: %+e\n", threadNum, -0.5*kali::log2Pi);
        //printf("computeLnPrior - threadNum: %d;                                 -log2(periodWidth*kali::Day)/kali::log2OfE: %+e\n", threadNum, -log2(periodWidth*kali::Day)/kali::log2OfE);
        //printf("computeLnPrior - threadNum: %d;                ((periodCenter*kali::Day) - period)/(periodWidth*kali::Day): %+e\n", threadNum, ((periodCenter*kali::Day) - period)/(periodWidth*kali::Day));
        //printf("computeLnPrior - threadNum: %d; -0.5*pow(((periodCenter*kali::Day) - period)/(periodWidth*kali::Day), 2.0): %+e\n", threadNum, -0.5*pow(((periodCenter*kali::Day) - period)/(periodWidth*kali::Day), 2.0));
        printf("computeLnPrior - threadNum: %d; periodPrior: %+e\n", threadNum, periodPrior);
        printf("computeLnPrior - threadNum: %d; totalFlux: %+e\n", threadNum, totalFlux);
        printf("computeLnPrior - threadNum: %d; fluxCenter: %+e\n", threadNum, fluxCenter);
        printf("computeLnPrior - threadNum: %d; fluxWidth: %+e\n", threadNum, fluxWidth);
        printf("computeLnPrior - threadNum: %d; totalFluxPrior: %+e\n", threadNum, totalFluxPrior);
	    printf("computeLnPrior - threadNum: %d; LnPrior: %+4.3e\n",threadNum,LnPrior);
	    printf("\n");
	#endif

	Data.currentLnPrior = LnPrior;

	return LnPrior;
	}

double kali::MBHB::computeLnLikelihood(LnLikeData *ptr2Data) {
	LnLikeData Data = *ptr2Data;

	int numCadences = Data.numCadences;
	double dt = kali::Day*Data.dt;
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
		//Contrib = mask[i]*(-1.0*log2(yerr[i])/kali::log2OfE -0.5*pow(((y[i] - totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()))/yerr[i]), 2.0));
		Contrib = mask[i]*(-1.0*log2(yerr[i])/kali::log2OfE -0.5*pow(((y[i] - totalFlux*getBeamingFactor2())/yerr[i]), 2.0));
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
			printf("-1.0*log2(yerr[i])/kali::log2OfE): %e\n",-1.0*log2(yerr[i])/kali::log2OfE);
			printf("Contrib: %e\n",Contrib);
		#endif
		LnLikelihood = LnLikelihood + Contrib;
		#ifdef DEBUG_COMPUTELNLIKELIHOOD
			printf("LnLikelihood: %e\n",LnLikelihood);
		#endif
		ptCounter += mask[i];
		}
	#ifdef DEBUG_COMPUTELNLIKELIHOOD
		printf("-0.5*ptCounter*kali::log2Pi: %e\n",-0.5*ptCounter*kali::log2Pi);
	#endif
	LnLikelihood += -0.5*ptCounter*kali::log2Pi;
	#ifdef DEBUG_COMPUTELNLIKELIHOOD
		printf("LnLikelihood: %e\n",LnLikelihood);
	#endif

	Data.cadenceNum = numCadences - 1;
	Data.currentLnLikelihood = LnLikelihood;
	return LnLikelihood;
	}

int kali::MBHB::Smoother(LnLikeData *ptr2Data, double *xSmooth) {
    LnLikeData Data = *ptr2Data;

	int numCadences = Data.numCadences;
	double *t = Data.t;

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
		xSmooth[i] = totalFlux*getBeamingFactor2();
		#ifdef DEBUG_SIMULATESYSTEM
			printf("totalFlux*getBeamingFactor2(): %e\n",totalFlux*getBeamingFactor2());
			//printf("totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()): %e\n",totalFlux*(1.0 - fracBeamedFlux + fracBeamedFlux*getBeamingFactor2()));
		#endif
		}
    return 0;
    }
