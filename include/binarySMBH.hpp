#ifndef BINARYSMBH_HPP
#define BINARYSMBH_HPP

#include <vector>

using namespace std;

double calcLnPrior(const vector<double> &x, vector<double>& grad, void* p2Args);

double calcLnPrior(double* walkerPos, void* vdPtr2LnLikeArgs);

double calcLnPosterior(const vector<double> &x, vector<double>& grad, void* p2Args);

double calcLnPosterior(double* walkerPos, void* vdPtr2LnLikeArgs);

double d2r(double degreeVal);
double r2d(double radianVal);

struct KeplersEqnData {
	double eccentricity;
	double M;
	};

double KeplerEqn(const vector<double> &x, vector<double> &grad, void *p2Data);

struct LnLikeData {
	int numCadences;
	double dt;
	int cadenceNum;
	double fracNoiseToSignal;
	double lowestFlux;
	double highestFlux;
	double currentLnPrior;
	double currentLnLikelihood;
	double currentLnPosterior;
	double *t;
	double *x;
	double *y;
	double *yerr;
	double *mask;
	double *lcX;
	double *lcP;
	};

class binarySMBH {
private:
	//double rPeriTot, rApoTot, a1, a2, rPeri1, rPeri2, rApo1, rApo2, m1, m2, rS1, rS2, totalMass, massRatio, reducedMass, period, eccentricity, eccentricityFactor, omega1, omega2, inclination, tau, alpha1, alpha2, epoch, M, E, nu, theta1, theta2, r1, r2, beta1, beta2, radialBeta1, radialBeta2, dF1, dF2, bF1, bF2, totalFlux, fracBeamedFlux, _radialBetaFactor1, _radialBetaFactor2;
	double rPeriTot, rApoTot, a1, a2, rPeri1, rPeri2, rApo1, rApo2, m1, m2, rS1, rS2, totalMass, massRatio, reducedMass, period, eccentricity, eccentricityFactor, omega1, omega2, inclination, tau, alpha1, alpha2, epoch, M, E, nu, theta1, theta2, r1, r2, beta1, beta2, radialBeta1, radialBeta2, dF1, dF2, bF1, bF2, totalFlux, _radialBetaFactor1, _radialBetaFactor2;
	void operator()();
public:
	binarySMBH();
	binarySMBH(double rPeriTot, double m1, double m2, double eccentricity, double omega, double inclination, double tau, double alpha1, double alpha2);
	int checkBinarySMBHParams(double *ThetaIn);
	void setBinarySMBH(double *ThetaIn);
	void setEpoch(double epochIn);
	double getEpoch();
	double getPeriod();
	double getA1();
	double getA2();
	double getRPeri1();
	double getRPeri2();
	double getRApo1();
	double getRApo2();
	double getRPeriTot();
	double getRApoTot();
	double getM1();
	double getM2();
	double getRS1();
	double getRS2();
	double getEccentricity();
	double getOmega1();
	double getOmega2();
	double getInclination();
	double getTau();
	double getMeanAnomoly();
	double getEccentricAnomoly();
	double getTrueAnomoly();
	double getR1();
	double getR2();
	double getTheta1();
	double getTheta2();
	double getBeta1();
	double getBeta2();
	double getRadialBeta1();
	double getRadialBeta2();
	double getDopplerFactor1();
	double getDopplerFactor2();
	double getBeamingFactor1();
	double getBeamingFactor2();
	double aH(double sigmaStars);
	double aGW(double sigmaStars, double rhoStars, double H);
	double durationInHardState(double sigmaStars, double rhoStars, double H);
	double ejectedMass(double sigmaStars, double rhoStars, double H);
	void print();

	void simulateSystem(LnLikeData *ptr2Data);
	void observeNoise(LnLikeData *ptr2Data, unsigned int noiseSeed, double* noiseRand);
	double computeLnLikelihood(LnLikeData *ptr2LnLikeData);
	double computeLnPrior(LnLikeData *ptr2LnLikeData);
};

struct LnLikeArgs {
	int numThreads;
	binarySMBH *Systems;
	LnLikeData *Data;
	};

#endif