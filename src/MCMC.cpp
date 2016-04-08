#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <algorithm>
#include <omp.h>
#include <string>
#include <fstream>
#include <iostream>
#include "MCMC.hpp"
//#include "CARMA.hpp"
#include "Constants.hpp"

//#define WRITE_ZS
//#define WRITE_WALKERS
//#define WRITE_MOVES

#if defined(WRITE_ZS) || defined (WRITE_WALKERS) || defined(WRITE_MOVES)
#include <stdio.h>
#include <unistd.h>
#include <fstream>
#include <string>
#endif

//#define DEBUG_INIT
//#define DEBUG_RUNMCMC
//#define DEBUG_RUNMCMC_DEEP
//#define DEBUG_CTORENSEMBLESAMPLER
//#define DEBUG_DTORENSEMBLESAMPLER

using namespace std;

EnsembleSampler::EnsembleSampler(int ndims, int nwalkers, int nsteps, int nthreads, double a, double (*func)(double* x, void* funcArgs), void* funcArgs, unsigned int zSeed, unsigned int bernoulliSeed, unsigned int walkerSeed) {
	#ifdef DEBUG_CTORENSEMBLESAMPLER
	printf("EnsembleSampler - Constructing obj at %p!\n",this);
	#endif

	/*!
	First, we make sure we have an even number of walkers. If not, we (silently!) increase the number of walkers by 1. Later, we will add code to make sure we have atleast twice the number of walkers as we have dimensions.
	*/
	/*if (nwalkers%2 == 1) {
		nwalkers += 1;
		}*/

	/*if ((nwalkers/2)%nthreads != 0) {
		nwalkers += 2*(nthreads-((nwalkers/2)%nthreads));
		}*/

	numDims = ndims;
	numWalkers = nwalkers;
	numSteps = nsteps;
	numThreads = nthreads;
	A = a;
	ZSeed = zSeed;
	WalkerSeed = walkerSeed;
	BernoulliSeed = bernoulliSeed;
	Func = func;
	FuncArgs = funcArgs;

	/*!
	We will store the MCMC result in Chain. Chain is laid out as follows - for each step, we store each dimension of each walker. Chain[dimNum + walkerNum*numDims + stepNum*numDims*numWalkers] contains the value of dimension dimNum of walker walkerNum at step stepNum. We calculate the size of the Chain required, sizeChain = numDims*numWalkers*numSteps, and then allocate space to hold Chain.
	*/
	int sizeChain = numDims*numWalkers*numSteps;
	int sizeStep = numDims*numWalkers;
	int sizeHalfStep = numDims*numWalkers/2;
	int halfNumWalkers = numWalkers/2;
	int numChoices = numWalkers*numSteps;

	Chain = static_cast<double*>(_mm_malloc(sizeChain*sizeof(double),64));

	/*!
	We need numZs = numWalkers*numSteps stretch factors, Z, to move numWalkers walkers over numSteps steps. After allocating the space required to hold Z, we use the vdRngBeta function from the Intel MKL VSL to populate Z. To use vdRngBeta, we create a VSL_BRNG_SFMT19937 basic random stream and then run vdRngBeta with p = 0.5, q = 1.0, a = 1/a, Beta = (a^2-1.0)/a.
	*/

	Zs = static_cast<double*>(_mm_malloc(numChoices*sizeof(double),64));
	WalkerChoice = static_cast<int*>(_mm_malloc(numChoices*sizeof(int),64));
	MoveYesNo = static_cast<int*>(_mm_malloc(numChoices*sizeof(int),64));
	LnLike = static_cast<double*>(_mm_malloc(numChoices*sizeof(double),64));

	for (int choiceNum = 0; choiceNum < numChoices; choiceNum++) {
		Zs[choiceNum] = 0.0;
		WalkerChoice[choiceNum] = 0;
		MoveYesNo[choiceNum] = 0;
		LnLike[choiceNum] = 0.0;
		}

	VSLStreamStatePtr ZStream, WalkerStream;
	vslNewStream(&ZStream, VSL_BRNG_SFMT19937, ZSeed);
	vslNewStream(&WalkerStream, VSL_BRNG_SFMT19937, WalkerSeed);
	vdRngBeta(VSL_RNG_METHOD_BETA_CJA_ACCURATE, ZStream, numChoices, Zs, 0.5, 1.0, 1.0/A, (pow(A,2.0)-1.0)/A);
	viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, WalkerStream, numChoices, WalkerChoice, 0, halfNumWalkers);
	vslDeleteStream(&ZStream);
	vslDeleteStream(&WalkerStream);

	/*!
	If the preprocessor macro WRITE_ZS is set in MCMC.cpp, we write the Zs out.
	*/
	#ifdef WRITE_ZS
	string ZSPath = "/home/exarkun/Desktop/Zs.dat";
	ofstream ZSFile;
	ZSFile.open(ZSPath);
	ZSFile.precision(16);
	for (int i = 0; i < numChoices-1; i++) {
		ZSFile << noshowpos << scientific << Zs[i] << endl;
		}
	ZSFile << noshowpos << scientific << Zs[numChoices-1];
	ZSFile.close();
	#endif

	/*!
	If the preprocessor macro WRITE_WALKERS is set in MCMC.cpp, we write the WalkerChoices out.
	*/
	#ifdef WRITE_WALKERS
	string WalkersPath = "/home/exarkun/Desktop/Walkers.dat";
	ofstream WalkersFile;
	WalkersFile.open(WalkersPath);
	WalkersFile.precision(16);
	for (int i = 0; i < numChoices-1; i++) {
		WalkersFile << noshowpos << scientific << WalkerChoice[i] << endl;
		}
	WalkersFile << noshowpos << scientific << WalkerChoice[numChoices-1];
	WalkersFile.close();
	#endif

	/*!
	We will have to pick 1 walker from the complimentary ensemble to move each walker from the current ensemble. We allocate an array, walkerChoice, to hold the indices of the random walkers picked from the complimentary ensemble. We also initialize a random stream that we shall later use to pick our walkers. We will have to make a decision about whether to move the current walker or not. We allocate moveYesNo to hold the result of the decision. We also initialize a random stream that we shall later use to make our choices.
	*/

	/*!
	We will store the log likelihoods in an array called LnLike. This way we do not have to re-compute the LnLike multiple times for the same point. LnLike[walkerNum + stepNum*numWalkers] holds the LnLike of walker walkerNum at step stepNum.
	*/ 

	/*!
	We will use numThreads number of pointers to access the old and new positions of the current walker(s) and the old position of the complimentary walker.
	*/
	}

EnsembleSampler::~EnsembleSampler() {
	#ifdef DEBUG_DTORENSEMBLESAMPLER
	printf("~EnsembleSampler - Freeing memory at %p!\n",this);
	#endif

	if (Chain) {
		_mm_free(Chain);
		Chain = nullptr;
		}

	if (Zs) {
		_mm_free(Zs);
		Zs = nullptr;
		}

	if (WalkerChoice) {
		_mm_free(WalkerChoice);
		WalkerChoice = nullptr;
		}

	if (MoveYesNo) {
		_mm_free(MoveYesNo);
		MoveYesNo = nullptr;
		}

	if (LnLike) {
		_mm_free(LnLike);
		LnLike = nullptr;
		}
	}

void EnsembleSampler::runMCMC(double* initPos) {

	/*!
	We begin by computing the LnLike values for our initial walker positions.
	*/
	#ifdef DEBUG_RUNMCMC
	printf("runMCMC - Starting runMCMC...\n");
	printf("runMCMC - Setting initial positions of walkers.\n");
	#endif

	int nsteps = numSteps;
	int nwalkers = numWalkers;
	int ndims = numDims;
	int nthreads = numThreads;

	#ifdef DEBUG_RUNMCMC
	printf("runMCMC - nsteps: %d\n",nsteps);
	printf("runMCMC - nwalkers: %d\n",nwalkers);
	printf("runMCMC - ndims: %d\n",ndims);
	printf("runMCMC - nthreads: %d\n",nthreads);
	#endif

	#ifdef DEBUG_RUNMCMC
	printf("runMCMC - numSteps: %d\n",numSteps);
	printf("runMCMC - numWalkers: %d\n",numWalkers);
	printf("runMCMC - numDims: %d\n",numDims);
	printf("runMCMC - numThreads: %d\n",numThreads);
	#endif

	int sizeChain = numDims*numWalkers*numSteps;
	int sizeStep = numDims*numWalkers;
	int sizeHalfStep = numDims*numWalkers/2;
	int halfNumWalkers = numWalkers/2;
	int numChoices = numWalkers*numSteps;

	#ifdef DEBUG_RUNMCMC
	printf("runMCMC - sizeChain: %d\n",sizeChain);
	printf("runMCMC - sizeStep: %d\n",sizeStep);
	printf("runMCMC - sizeHalfStep: %d\n",sizeHalfStep);
	printf("runMCMC - halfNumWalkers: %d\n",halfNumWalkers);
	printf("runMCMC - numChoices: %d\n",numChoices);
	#endif

	double *p2Chain = &Chain[0], *p2LnLike = &LnLike[0], *p2Zs = &Zs[0]; 
	int *p2WalkerChoice = &WalkerChoice[0], *p2MoveYesNo = &MoveYesNo[0];

	double (*p2Func)(double* x, void* FuncArgs) = Func;
	void* p2FuncArgs = FuncArgs;

	#ifdef DEBUG_RUNMCMC_DEEP
	int threadNum = omp_get_thread_num();
	for (int walkerNum = 0; walkerNum < numWalkers; walkerNum++) {
		printf("runMCMC - Thread: %d; walkerNum: %d\n",threadNum,walkerNum);
		printf("runMCMC - Thread: %d; initPos: %f\n",threadNum,initPos[walkerNum*numDims]);
		fflush(0);
		}
	#endif

	#pragma omp parallel for default(none) shared(nwalkers,ndims,nthreads,sizeChain,sizeStep,sizeHalfStep,halfNumWalkers,p2Chain,p2LnLike,p2Func,p2FuncArgs,initPos)// num_threads(numThreads)
	for (int walkerNum = 0; walkerNum < nwalkers; walkerNum++) {

		#ifdef DEBUG_RUNMCMC_OMP
		printf("numThreads: %d\n",numThreads);
		printf("omp_num_threads(): %d\n",omp_get_num_threads());
		printf("omp_get_thread_num(): %d\n",omp_get_thread_num());
		fflush(0);
		#endif

		double *currWalkerNewPos = nullptr;
		int threadNum = omp_get_thread_num();

		#ifdef DEBUG_RUNMCMC
		printf("runMCMC - Thread: %d; Walker: %d\n", threadNum, walkerNum);
		#endif

		for (int dimNum = 0; dimNum < ndims; dimNum++) {
			p2Chain[dimNum + walkerNum*ndims] = initPos[dimNum + walkerNum*ndims];

			#ifdef DEBUG_RUNMCMC_DEEP
			printf("runMCMC - Thread: %d; Walker: %d; Dim: %d; Val: %f\n", threadNum, walkerNum,dimNum,p2Chain[dimNum + walkerNum*ndims]);
			#endif
			}

		currWalkerNewPos = &p2Chain[walkerNum*ndims];

		#ifdef DEBUG_RUNMCMC_DEEP
		printf("runMCMC - Thread: %d; walkerNum: %d\n",threadNum,walkerNum);
		printf("runMCMC - Thread: %d; currWalkerNewPos: %f\n",threadNum,currWalkerNewPos[0]);
		fflush(0);
		#endif

		p2LnLike[walkerNum] = p2Func(currWalkerNewPos, p2FuncArgs);

		#ifdef DEBUG_RUNMCMC
		printf("runMCMC - Thread: %d; LnLike[%d]: %f\n",threadNum,walkerNum,p2LnLike[walkerNum]);
		printf("\n");
		#endif
		}

	#ifdef DEBUG_RUNMCMC
	printf("\n");
	#endif

	double *currSubSetOld = nullptr, *compSubSetOld = nullptr, *currSubSetNew = nullptr;
	unsigned int bernoulliSeed = BernoulliSeed;

	#ifdef DEBUG_RUNMCMC
	printf("runMCMC - Setting Bernoulli Stream...\n");
	#endif

	VSLStreamStatePtr* BernoulliStream = (VSLStreamStatePtr*)_mm_malloc(nthreads*sizeof(VSLStreamStatePtr),64);
	//vslNewStream(&BernoulliStream[0],VSL_BRNG_MCG59,BernoulliSeed);
	//vslNewStream(&BernoulliStream[0],VSL_BRNG_WH,BernoulliSeed);
	#pragma omp parallel for default(none) shared(nthreads,BernoulliStream,bernoulliSeed,halfNumWalkers)
	for (int threadNum = 0; threadNum < nthreads; threadNum++) {
		vslNewStream(&BernoulliStream[threadNum], VSL_BRNG_SFMT19937, bernoulliSeed);
		//vslNewStream(&BernoulliStream[walkerNum], VSL_BRNG_WH, BernoulliSeed);
		//vslCopyStream(&BernoulliStream[threadNum],BernoulliStream[0]);
		vslSkipAheadStream(BernoulliStream[threadNum], threadNum*(halfNumWalkers/nthreads));
		//vslLeapfrogStream(BernoulliStream[threadNum],threadNum,nthreads);
		}

	#ifdef DEBUG_RUNMCMC
	printf("runMCMC - Starting MCMC...\n");
	#endif

	/*! We first run a loop over all the steps. Recall that the 0th step is the starting step and we don't want to do anything for that step. As before, k keeps track of the current step.
	*/
	for (int stepNum = 1; stepNum < numSteps; stepNum++) {

		/*! To enable parallelization, we split our walkers into two subsets indexed by 0 and 1. We will move all the walkers in the current subset, currSubSet, based on randomly chosen walkers in the complimentary subset, compSubSet. We index the subsets using l.
		*/
		for (int subSetNum = 0; subSetNum < 2; subSetNum++) {

			/*! We set currSubSet to point to the current subset and set compSubSet to point to the complimentary subset.
			If subSetNum = 0, we want 1*sizeHalfStep.
			If subSetNum = 1, we want 0*sizeHalfStep.
			Use ((l+1)%2).
			*/

			currSubSetOld = &Chain[(stepNum-1)*sizeStep + subSetNum*sizeHalfStep];
			compSubSetOld = &Chain[(stepNum-1)*sizeStep + ((subSetNum+1)%2)*sizeHalfStep];
			currSubSetNew = &Chain[stepNum*sizeStep + subSetNum*sizeHalfStep];

			/*! 
			Move over walkers in current sub-chain
			*/
			#pragma omp parallel for default(none) shared(stepNum,subSetNum,log2OfE,nwalkers,ndims,nthreads,sizeChain,sizeStep,sizeHalfStep,halfNumWalkers,p2Chain,p2LnLike,p2Func,p2FuncArgs,currSubSetOld,compSubSetOld,currSubSetNew,p2Zs,p2WalkerChoice,p2MoveYesNo,BernoulliStream)// num_threads(numThreads)
			for (int walkerNum = 0; walkerNum < halfNumWalkers; walkerNum++) {

				#ifdef DEBUG_RUNMCMC_OMP
				printf("numThreads: %d\n",numThreads);
				printf("omp_num_threads(): %d\n",omp_get_num_threads());
				printf("omp_get_thread_num(): %d\n",omp_get_thread_num());
				fflush(0);
				#endif

				int threadNum = omp_get_thread_num();
				double newLnLike = 0.0, oldLnLike = 0.0, pAccept = 0.0;
				double *compWalkerOldPos = nullptr, *currWalkerOldPos = nullptr, *currWalkerNewPos = nullptr;

				/*!
				First we get the old position of the current walker.
				*/
				currWalkerOldPos = &currSubSetOld[walkerNum*ndims];
				//printf("stepNum: %d; walkerNum: %d; threadNum: %d; Address of currWalkerOldPos: %p\n",stepNum,walkerNum,threadNum,currWalkerOldPos);

				#ifdef DEBUG_RUNMCMC
				printf("runMCMC - threadNum: %d; stepNum: %d; currWalkerNum: %d; Index: %d\n",threadNum,stepNum,walkerNum+halfNumWalkers*subSetNum,(stepNum-1)*numWalkers + subSetNum*halfNumWalkers + walkerNum);
				printf("runMCMC - threadNum: %d; stepNum: %d; currWalkerNum: %d; Old Location: ",threadNum,stepNum,walkerNum+halfNumWalkers*subSetNum);
				for (int i = 0; i < numDims; i++) {
					printf("%f ",currWalkerOldPos[i]);
					}
				printf("\n");
				#endif

				/*!
				Pick walker from complimentary ensemble and get the old position of that walker.
				*/

				compWalkerOldPos = &compSubSetOld[p2WalkerChoice[(stepNum-1)*nwalkers + subSetNum*halfNumWalkers + walkerNum]*ndims];

				#ifdef DEBUG_RUNMCMC
				printf("runMCMC - threadNum: %d; stepNum: %d; compWalkerNum: %d; Old Location: ",threadNum,stepNum,p2WalkerChoice[(stepNum-1)*numWalkers + subSetNum*halfNumWalkers + walkerNum]+halfNumWalkers*((subSetNum+1)%2));
				for (int i = 0; i < numDims; i++) {
					printf("%f ",compWalkerOldPos[i]);
					}
				printf("\n");
				#endif

				/*!
				Now we get the location of the new position of the current walker.
				*/
				currWalkerNewPos = &currSubSetNew[walkerNum*ndims];

				/*!
				Calculate the (tentative) new location to walk to.
				*/
				#ifdef DEBUG_RUNMCMC
				printf("runMCMC - threadNum: %d; stepNum: %d; currWalkerNum: %d; Z: %f\n",threadNum,stepNum,walkerNum+halfNumWalkers*subSetNum,Zs[(stepNum-1)*numWalkers + subSetNum*halfNumWalkers + walkerNum]);
				#endif

				for (int dimNum = 0; dimNum < ndims; dimNum++) {
					currWalkerNewPos[dimNum] = compWalkerOldPos[dimNum] + p2Zs[(stepNum-1)*nwalkers + subSetNum*halfNumWalkers + walkerNum]*(currWalkerOldPos[dimNum] - compWalkerOldPos[dimNum]);
					}

				#ifdef DEBUG_RUNMCMC
				printf("runMCMC - threadNum: %d; stepNum: %d; currWalkerNum: %d; New Location: ",threadNum,stepNum,walkerNum+halfNumWalkers*subSetNum);
				for (int i = 0; i < ndims; i++) {
					printf("%f ",currWalkerNewPos[i]);
					}
				printf("\n");
				#endif

				/*!
				Now compute the logLike at the new location and fetch the LnLike at the old location.
				*/
				newLnLike = p2Func(currWalkerNewPos, p2FuncArgs);
				oldLnLike = p2LnLike[walkerNum + subSetNum*halfNumWalkers + (stepNum-1)*nwalkers];
				//oldLnLike = p2Func(currWalkerOldPos, p2FuncArgs);

				#ifdef DEBUG_RUNMCMC
				printf("runMCMC - threadNum: %d; stepNum: %d; currWalkerNum: %d; Old LnLike: %f\n",threadNum,stepNum,walkerNum+halfNumWalkers*subSetNum,oldLnLike);
				printf("runMCMC - threadNum: %d; stepNum: %d; currWalkerNum: %d; New LnLike: %f\n",threadNum,stepNum,walkerNum+halfNumWalkers*subSetNum,newLnLike);
				#endif

				/*!
				Calculate likelihood of accepting proposal. If both log likelihoods are non-neg infinity, calculate it. If the new likelihood is 
				*/
				if ((oldLnLike != -HUGE_VAL) and (newLnLike != -HUGE_VAL)) {
					pAccept = exp(min(0.0, (ndims-1)*(log2(p2Zs[(stepNum-1)*nwalkers + subSetNum*halfNumWalkers + walkerNum])/log2OfE) + newLnLike - oldLnLike));
					} else if ((oldLnLike == -HUGE_VAL) and (newLnLike != -HUGE_VAL)) {
					pAccept = 1.0;
					} else if ((oldLnLike != -HUGE_VAL) and (newLnLike == -HUGE_VAL)) {
					pAccept = 0.0;
					} else if ((oldLnLike == -HUGE_VAL) and (newLnLike == -HUGE_VAL)) {
					pAccept = 0.0;
					}

				#ifdef DEBUG_RUNMCMC
				printf("runMCMC - threadNum: %d; stepNum: %d;  currWalkerNum: %d; pAccept: %f\n",threadNum,stepNum,walkerNum+halfNumWalkers*subSetNum,pAccept);
				#endif

				/*!
				Actually do a coin toss to test the proposal.
				*/
				viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, BernoulliStream[threadNum], 1, &p2MoveYesNo[(stepNum-1)*nwalkers + subSetNum*halfNumWalkers + walkerNum], pAccept);


				#ifdef DEBUG_RUNMCMC
				printf("runMCMC - threadNum: %d; stepNum: %d;  currWalkerNum: %d; moveYesNo: %d\n",threadNum,stepNum,walkerNum+halfNumWalkers*subSetNum,p2MoveYesNo[(stepNum-1)*nwalkers + subSetNum*halfNumWalkers + walkerNum]);
				#endif

				/*!
				Check the result of the coin toss. Based on the result, either move the walker, or leave it alone. Write out the LnLike to the correct location.
				*/
				if (p2MoveYesNo[(stepNum-1)*nwalkers + subSetNum*halfNumWalkers + walkerNum] == 1) { // Record the new LnLike as the LnLike for this walker. He has already moved so do nothing to his position.
					p2LnLike[walkerNum + subSetNum*halfNumWalkers + stepNum*nwalkers] = newLnLike;
					} else { // Record the old LnLike as the LnLike for this walker. Move the walker's position back.
					p2LnLike[walkerNum + subSetNum*halfNumWalkers + stepNum*nwalkers] = oldLnLike;
					for (int dimNum = 0; dimNum < ndims; dimNum++) {
						currWalkerNewPos[dimNum] = currWalkerOldPos[dimNum];
						}
					}

				#ifdef DEBUG_RUNMCMC
				printf("runMCMC - threadNum: %d; stepNum: %d; currWalkerNum: %d; New Location: ",threadNum,stepNum,walkerNum+halfNumWalkers*subSetNum);
				for (int i = 0; i < numDims; i++) {
					printf("%f ",currWalkerNewPos[i]);
					}
				printf("\n");
				printf("\n");
				#endif

				}
			}

		#pragma omp parallel for default(none) shared(nthreads,BernoulliStream,bernoulliSeed,halfNumWalkers)
		for (int threadNum = 0; threadNum < nthreads; threadNum++) {
			vslSkipAheadStream(BernoulliStream[threadNum], (nthreads-1)*(halfNumWalkers/nthreads));
			}

		}

	#ifdef WRITE_MOVES
	string ChoicesPath = "/home/exarkun/Desktop/Choices.dat";
	ofstream ChoicesFile;
	ChoicesFile.open(ChoicesPath);
	ChoicesFile.precision(16);
	for (int i = 0; i < numChoices-1; i++) {
		ChoicesFile << noshowpos << scientific << MoveYesNo[i] << endl;
		}
	ChoicesFile << noshowpos << scientific << MoveYesNo[numChoices-1];
	ChoicesFile.close();
	#endif

	for (int threadNum = 0; threadNum < nthreads; threadNum++) {
		vslDeleteStream(&BernoulliStream[threadNum]);
		}
	_mm_free(BernoulliStream);
	}

void EnsembleSampler::getChain(double *ChainPtr) {
	int nsteps = numSteps;
	int nwalkers = numWalkers;
	int ndims = numDims;
	int sizeChain = numDims*numWalkers*numSteps;
	double* Ptr2Chain = &Chain[0];
	#pragma omp parallel for simd default(none) shared(sizeChain, ChainPtr, Ptr2Chain)
	for (int i = 0; i < sizeChain; ++i) {
		ChainPtr[i] = Ptr2Chain[i];
		}
	}

void EnsembleSampler::getLnLike(double *LnLikePtr) {
	int nsteps = numSteps;
	int nwalkers = numWalkers;
	int ndims = numDims;
	int sizeLnLike = numWalkers*numSteps;
	double* Ptr2LnLike = &LnLike[0];
	#pragma omp parallel for simd default(none) shared(sizeLnLike, LnLikePtr, Ptr2LnLike)
	for (int i = 0; i < sizeLnLike; ++i) {
		LnLikePtr[i] = Ptr2LnLike[i];
		}
	}