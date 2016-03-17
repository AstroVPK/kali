#ifndef BINARYSMBH_HPP
#define BINARY_HPP

#include <vector>

using namespace std;

namespace CARMA {

	double d2r(double degreeVal);
	double r2d(double radianVal);

	struct KeplersEqnData {
		double ellipticity;
		double M;
		};

	double KeplerEqn(const vector<double> &x, vector<double> &grad, void *p2Data);

	class binarySMBH {
	private:
		double a1, a2, m1, m2, totalMass, massRatio, reducedMass, period, ellipticity, ellipticityFactor, omega1, omega2, inclination, tau, alpha1, alpha2, t, M, E, nu, theta1, theta2, r1, r2, beta1, beta2, radialBeta1, radialBeta2, dF1, dF2, bF1, bF2;
	public:
		binarySMBH();
		binarySMBH(double a1, double a2, double m1, double m2, double ellipticity, double omega, double inclination, double tau, double alpha1, double alpha2);
		void operator()(double epoch);
		void setEpoch(double t);
		double getEpoch();
		double getPeriod();
		double getA1();
		double getA2();
		double getEllipticity();
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
	};
}

#endif