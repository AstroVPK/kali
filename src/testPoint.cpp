#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "Acquire.hpp"
#include "CARMA.hpp"
#include "MCMC.hpp"

#define TIME_LNLIKE
#define TIME_MCMC

using namespace std;

int main() {
	cout.clear();
	cout << endl;
	cout << "Program: testPoint" << endl;
	cout << "Purpose: Program to test individual points in C-ARMA parameter space." << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institution: University of Pennsylvania, Department of Physics & Astronomy" << endl;
	cout << "Email: vishal.kasliwal@gmail.com" << endl;
	cout << endl;

	string basePath;
	AcquireDirectory(cout,cin,"Full path to CARMA directory: ","Invalid path!\n",basePath);
	basePath += "/";
	cout << "CARMA directory: " << basePath << endl;

	string yFilePath = basePath+"y.dat";
	ifstream yFile;
	yFile.open(yFilePath);
	string yLine;
	getline(yFile,yLine);
	istringstream yRecord(yLine);
	string throwAway;
	yRecord >> throwAway;
	cout << "throwAway: " << throwAway << endl;
	int numPts;
	yRecord >> numPts;
	cout << "numPts: " << numPts << endl;
	double* y = static_cast<double*>(_mm_malloc(numPts*sizeof(double),64));
	double* yerr = static_cast<double*>(_mm_malloc(numPts*sizeof(double),64));
	double* mask = static_cast<double*>(_mm_malloc(numPts*sizeof(double),64));
	for (int lineCtr = 0; lineCtr < numPts; ++lineCtr) {
		getline(yFile,yLine);
		istringstream yRecord(yLine);
		yRecord >> y[lineCtr];
		yRecord >> yerr[lineCtr];
		if (y[lineCtr] != 0.0) {
			mask[lineCtr] = 1.0;
			} else {
			mask[lineCtr] = 0.0;
			}
		}

	for (int lineCtr = 0; lineCtr < numPts; ++lineCtr) {
		cout << "Cad: " << lineCtr << "; y[" << lineCtr << "]: " << y[lineCtr] << "; yerr[" << lineCtr << "]: " << yerr[lineCtr] << "; mask[" << lineCtr << "]: " << mask[lineCtr] << endl;
		}

	cout << "y has been read in" << endl;

	cout << "Check the parameter values of an C-ARMA light curve with p C-AR and q C-MA co-efficients." << endl;
	int pMaster = 0, qMaster = 0;
	AcquireInput(cout,cin,"Number of AR coefficients p: ","Invalid value.\n",pMaster);
	qMaster = pMaster;
	while (qMaster >= pMaster) {
		cout << "The number of MA coefficients (q) must be less than the number of AR coefficients (p) if the system is to \n correspond to a C-ARMA process." << endl;
		cout << "Please select q < " << pMaster << endl;
		AcquireInput(cout,cin,"Number of MA coefficients q: ","Invalid value.\n",qMaster);
		}
	cout << "Creating C-ARMA model with " << pMaster << " AR components and " << qMaster << " MA components." << endl;
	CARMA SystemMaster = CARMA();
	SystemMaster.allocCARMA(pMaster, qMaster);
	cout << "Allocated " << SystemMaster.get_allocated() << " bytes for the C-ARMA model!" << endl;

	double* ThetaMaster = static_cast<double*>(_mm_malloc((pMaster+qMaster+1)*sizeof(double),64));
	#pragma omp parallel for simd default(none) shared(pMaster,qMaster,ThetaMaster)
	for (int i = 0; i < pMaster+qMaster+1; i++) {
		ThetaMaster[i] = 0.0;
		}

	int doneYN = 0, goodYN = 0;
	double LnLike = 0.0;
	string dummy;
	getline(cin, dummy);
	string inputString;
	while (doneYN == 0) {
		cout << endl;
		cout << endl;

		cout << "Values of the C-ARMA parameters to be tested: ";
		/*getline(cin,inputString,'\n');
		cin.ignore();
		istringstream inputRecord(inputString);
		for (int dimNum = 0; dimNum < pMaster+qMaster+1; ++dimNum) {
			inputRecord >> ThetaMaster[dimNum];
			}*/

		double t_incr = 0.0;
		cout << "Set the sampling length t_incr such that t_incr > 0.0" << endl;
		AcquireInput(cout,cin,"Set the value of t_incr: ","Invalid value.\n",t_incr);

		for (int i = 0; i < pMaster; i++) {
			cout << "Set the value of a_" << i+1;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		AcquireInput(cout,cin,"Set the value of b_0: ","Invalid value.\n",ThetaMaster[pMaster]);

		for (int i = 1+pMaster; i < 1+pMaster+qMaster; i++) {
			cout << "Set the value of b_" << i-pMaster;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		cout << "System is set to use the following parameters..." << endl;
		for (int i = 0; i < pMaster; i++) {
			cout << "a_" << i+1 << ": " << ThetaMaster[i] << endl;
			}
		cout << "b_0: " << ThetaMaster[pMaster] << endl;
		for (int i = 1 + pMaster; i < 1 + pMaster + qMaster; i++) {
			cout << "b_" << i - pMaster << ": " << ThetaMaster[i] << endl;
			}
		cout << endl;

		cout << endl;
		cout << "Checking to see if the C-ARMA parameters are good, & computing LnLike" << endl;
		goodYN = SystemMaster.checkCARMAParams(ThetaMaster);
		cout << "System parameters are ";
		if (goodYN == 0) {
			cout << "bad!" << endl;
			LnLike = -HUGE_VAL;
			cout << endl;
			} else {
			cout << "good!" << endl;
			SystemMaster.setCARMA(ThetaMaster);
			SystemMaster.set_t(t_incr);
			SystemMaster.solveCARMA();
			SystemMaster.resetState();
			LnLike = SystemMaster.computeLnLike(numPts, y, yerr, mask);
			cout << endl;
			}

		cout.precision(16);
		cout << "LnLike: " << fixed << noshowpos << LnLike << endl;
		cout << endl;
		cout << endl;

		AcquireInput(cout,cin,"Done? (1/0): ","Input not parsed!\n",doneYN);
		}

	SystemMaster.deallocCARMA();
	_mm_free(ThetaMaster);
	_mm_free(y);
	_mm_free(yerr);
	_mm_free(mask);
	}