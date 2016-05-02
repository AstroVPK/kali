#include <mathimf.h>
#include <complex>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include "LC.hpp"

using namespace std;

	LCData::LCData() {
		numCadences = 0;
		dt = 0.0;
		tolIR = 0.0;
		fracIntrinsicVar = 0.0;
		fracNoiseToSignal = 0.0;
		maxSigma = 0.0;
		minTimescale = 0.0;
		maxTimescale = 0.0;
		t = nullptr;
		x = nullptr;
		y = nullptr;
		yerr = nullptr;
		mask = nullptr;
		lcXSim = nullptr;
		lcPSim = nullptr;
		lcXComp = nullptr;
		lcPComp = nullptr;
	}