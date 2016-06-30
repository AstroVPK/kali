#ifndef BINARYSMBH_HPP
#define BINARYSMBH_HPP

#include <vector>

using namespace std;

double d2r(double degreeVal);
double r2d(double radianVal);

struct KeplersEqnData {
	double ellipticity;
	double M;
	};

double KeplerEqn(const vector<double> &x, vector<double> &grad, void *p2Data);

struct LnLikeData {
	int numCadences;
	int cadenceNum;
	double tolIR;
	double t_incr;
	double fracIntrinsicVar;
	double fracNoiseToSignal;
	double maxSigma;
	double minTimescale;
	double maxTimescale;
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
	double rPer, a1, a2, m1, m2, rS1, rS2, totalMass, massRatio, reducedMass, period, ellipticity, ellipticityFactor, omega1, omega2, inclination, tau, alpha1, alpha2, t, M, E, nu, theta1, theta2, r1, r2, beta1, beta2, radialBeta1, radialBeta2, dF1, dF2, bF1, bF2, totalFlux, fracBeamedFlux;
public:
	binarySMBH();
	binarySMBH(double rPer, double m1, double m2, double ellipticity, double omega, double inclination, double tau, double alpha1, double alpha2);
	void operator()(double epoch);
	int checkBinarySMBHParams(double *ThetaIn);
	void setBinarySMBH(double *ThetaIn);
	void setEpoch(double tIn);
	double getEpoch();
	double getPeriod();
	double getA1();
	double getA2();
	double getM1();
	double getM2();
	double getRS1();
	double getRS2();
	double getEllipticity();
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
};

#endif