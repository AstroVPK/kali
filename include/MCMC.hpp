#ifndef MCMC_HPP
#define MCMC_HPP

#include <string>
//#include "Kalman.hpp"

using namespace std;

class EnsembleSampler {
private:
	int numDims, numWalkers, numSteps, numThreads;
	unsigned int ZSeed, BernoulliSeed, WalkerSeed;
	double A;//, newLnLike, oldLnLike, pAccept;
	//VSLStreamStatePtr ZStream, WalkerStream, *BernoulliStream;
	double *Chain, *Zs, *LnLike;
	//double *currSubSetOld, *compSubSetOld, *currSubSetNew;
	//double **compWalkerOldPos, **currWalkerOldPos, **currWalkerNewPos;
	int *WalkerChoice, *MoveYesNo;
	double (*Func)(double* x, void* FuncArgs);
	void* FuncArgs;
public:
	EnsembleSampler(int ndims, int nwalkers, int nsteps, int nthreads, double a, double (*func)(double* x, void* funcArgs), void* funcArgs, unsigned int zSeed, unsigned int bernoulliSeed, unsigned int walkerSeed);
	~EnsembleSampler();
	void runMCMC(double* initPos);
	void writeChain(string filePath, int mode);
	};

#endif