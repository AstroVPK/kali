#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

/*#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include <nlopt.hpp>
#include "CARMA.hpp"
#include "MCMC.hpp"
#include "Constants.hpp"
#include "rdrand.hpp"*/

using namespace std;

int getRandomsCARMA(int numRequested, unsigned int *Randoms);

int testSystemCARMA(double dt, int p, int q, double *Theta);

int printSystemCARMA(double dt, int p, int q, double *Theta);

int makeIntrinsicLCCARMA(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, int *cadence, double *mask, double *t, double *x);

double getMeanFluxCARMA(int p, int q, double *Theta, double fracIntrinsicVar);

int makeObservedLCCARMA(double dt, int p, int q, double *Theta, bool IR, double tolIR, double fracIntrinsicVar, double fracSignalToNoise, int numBurn, int numCadences, int startCadence, unsigned int burnSeed, unsigned int distSeed, unsigned int noiseSeed, int *cadence, double *mask, double *t, double *y, double *yerr);

double computeLnLikelihoodCARMA(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr);

double computeLnPosteriorCARMA(double dt, int p, int q, double *Theta, bool IR, double tolIR, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale);

int fitCARMACARMA(double dt, int p, int q, bool IR, double tolIR, double scatterFactor, int numCadences, int *cadence, double *mask, double *t, double *y, double *yerr, double maxSigma, double minTimescale, double maxTimescale, int nthreads, int nwalkers, int nsteps, int maxEvals, double xTol, unsigned int zSSeed, unsigned int walkerSeed, unsigned int moveSeed, unsigned int xSeed, double* xStart, double *Chain, double *LnPosterior);

#endif