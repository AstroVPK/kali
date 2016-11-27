#ifndef MBHB_HPP
#define MBHB_HPP

#include <vector>

using namespace std;

namespace kali {

double calcLnPrior(const vector<double> &x, vector<double>& grad, void* p2Args);

double calcLnPrior(double* walkerPos, void* vdPtr2LnLikeArgs);

double calcLnPosterior(const vector<double> &x, vector<double>& grad, void* p2Args);

double calcLnPosterior(double* walkerPos, void* vdPtr2LnLikeArgs, double & LnPrior, double &LnLikelihood);

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

class MBHB {
private:
	double rPeribothronTot, rApobothronTot, a1, a2, rPeribothron1, rPeribothron2, rApobothron1, rApobothron2, m1, m2, rS1, rS2, totalMass, massRatio, reducedMass, period, eccentricity, eccentricityFactor, omega1, omega2, inclination, tau, alpha1, alpha2, epoch, M, E, nu, theta1, theta2, r1, r2, beta1, beta2, radialBeta1, radialBeta2, dF1, dF2, bF1, bF2, totalFlux, _radialBetaFactor1, _radialBetaFactor2;
	void operator()();
public:
	MBHB();
	MBHB(double a1Val, double a2Val, double periodVal, double eccentricity, double omega, double inclination, double tau, double alpha1, double alpha2);
	int checkMBHBParams(double *ThetaIn);
	void setMBHB(double *ThetaIn);
	void setEpoch(double epochIn);
	double getEpoch();
	double getPeriod();
	double getA1();
	double getA2();
	double getRPeribothron1();
	double getRPeribothron2();
	double getRApobothron1();
	double getRApobothron2();
	double getRPeribothronTot();
	double getRApobothronTot();
	double getM1();
	double getM2();
	double getM12();
	double getM2OverM1();
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
	MBHB *Systems;
	LnLikeData *Data;
	};

} // namespace kali

#endif
