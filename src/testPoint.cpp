#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "Acquire.hpp"
#include "Kalman.hpp"
#include "MCMC.hpp"

#define TIME_LNLIKE
#define TIME_MCMC

using namespace std;

int main() {
	cout.clear();
	cout << endl;
	cout << "Program: testPoint" << endl;
	cout << "Purpose: Program to test individual oints in C-ARMA parameter space." << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institution: Drexel university, Department of Physics" << endl;
	cout << "Email: vpk24@drexel.edu" << endl;
	cout << endl;

	string basePath;
	AcquireDirectory(cout,cin,"Full path to output directory: ","Invalid path!\n",basePath);
	basePath += "/";
	cout << "Output directory: " << basePath << endl;

	string yFilePath = basePath+"y.dat";
	ifstream yFile;
	yFile.open(yFilePath);
	string yLine;
	getline(yFile,yLine);
	istringstream yRecord(yLine);
	string throwAway;
	yRecord >> throwAway;
	int numPts;
	yRecord >> numPts;
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
			}
		}

	cout << "y has been read in" << endl;

	cout << "Check the parameter values on an ARMA light curve with p AR and q MA co-efficients." << endl;
	int pMaster = 0, qMaster = 0;
	AcquireInput(cout,cin,"Number of AR coefficients p: ","Invalid value.\n",pMaster);
	qMaster = pMaster;
	while (qMaster >= pMaster) {
		cout << "The number of MA coefficients (q) must be less than the number of AR coefficients (p) if the system is to \n correspond to a C-ARMA process." << endl;
		cout << "Please select q < " << pMaster << endl;
		AcquireInput(cout,cin,"Number of MA coefficients q: ","Invalid value.\n",qMaster);
		}
	cout << "Creating DLM with " << pMaster << " AR components and " << qMaster << " MA components." << endl;
	DLM SystemMaster = DLM();
	SystemMaster.allocDLM(pMaster, qMaster);
	cout << "Allocated " << SystemMaster.allocated << " bytes for the DLM!" << endl;

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

		cout << "Values of the DLM parameters to be tested: ";
		/*getline(cin,inputString,'\n');
		cin.ignore();
		istringstream inputRecord(inputString);
		for (int dimNum = 0; dimNum < pMaster+qMaster+1; ++dimNum) {
			inputRecord >> ThetaMaster[dimNum];
			}*/

		cout << "Set the standard deviation of the disturbances (sigma_dist) such that sigma_dist > 0.0" << endl;
		AcquireInput(cout,cin,"Set the value of sigma_dist: ","Invalid value.\n",ThetaMaster[0]);

		for (int i = 1; i < 1+pMaster; i++) {
			cout << "Set the value of phi_" << i;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		for (int i = 1+pMaster; i < 1+pMaster+qMaster; i++) {
			cout << "Set the value of theta_" << i-pMaster;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		cout << "System is set to use the following parameters..." << endl;
		cout << "sigma_dist: " << ThetaMaster[0] << endl;
		for (int i = 1; i < 1+pMaster; i++) {
			cout << "phi_" << i << ": " << ThetaMaster[i] << endl;
			}
		for (int i = 1+pMaster; i < 1+pMaster+qMaster; i++) {
			cout << "theta_" << i << ": " << ThetaMaster[i] << endl;
			}
		cout << endl;

		SystemMaster.setDLM(ThetaMaster);
		SystemMaster.resetState();

		cout << endl;
		cout << "Checking to see if the system is stable, invertible, not-redundant, and reasonable, & computing LnLike" << endl;
		goodYN = SystemMaster.checkARMAParams(ThetaMaster);
		cout << "System parameters are ";
		if (goodYN == 0) {
			cout << "bad!" << endl;
			LnLike = -HUGE_VAL;
			cout << endl;
			} else {
			cout << "good!" << endl;
			LnLike = SystemMaster.computeLnLike(numPts, y, yerr, mask);
			cout << endl;
			}

		cout.precision(16);
		cout << "LnLike: " << fixed << noshowpos << LnLike << endl;
		cout << endl;
		cout << endl;

		AcquireInput(cout,cin,"Done? (1/0): ","Input not parsed!\n",doneYN);
		}

	SystemMaster.deallocDLM();
	_mm_free(ThetaMaster);
	_mm_free(y);
	_mm_free(yerr);
	_mm_free(mask);
	}