#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <string>
#include <sstream>
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
	cout << "Institutions: University of Pennsylvania (Department of Physics & Astronomy) & Princeton University (Department of Astrophysical Sciences)" << endl;
	cout << "Email: vishal.kasliwal@gmail.com" << endl;
	cout << endl;

	string basePath;
	vector<string> word(2);
	AcquireDirectory(cout,cin,"Full path to CARMA directory: ","Invalid path!\n",basePath);
	basePath += "/";
	cout << "CARMA directory: " << basePath << endl;

	string line, yFilePath = basePath+"y.dat";
	ifstream yFile;
	yFile.open(yFilePath);

	getline(yFile,line);
	istringstream record1(line);
	for (int i = 0; i < 2; ++i) {
		record1 >> word[i];
		}
	int numCadences = stoi(word[1]);
	cout << "numCadences: " << numCadences << endl;

	getline(yFile,line);
	istringstream record2(line);
	for (int i = 0; i < 2; ++i) {
		record2 >> word[i];
		}
	int numObservations = stoi(word[1]);
	cout << "numObservations: " << numObservations << endl;

	getline(yFile,line);
	istringstream record3(line);
	for (int i = 0; i < 2; ++i) {
		record3 >> word[i];
		}
	double meanFlux = stod(word[1]);
	cout << "meanFlux: " << meanFlux << endl;

	int* cadence = static_cast<int*>(_mm_malloc(numCadences*sizeof(double),64));
	double* mask = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	double* y = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));
	double* yerr = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));

	vector<string> wordNew(4);
	string lineNew;
	istringstream recordNew;
	cout.precision(16);
	int i = 0;
	while (!yFile.eof()) {
		getline(yFile,lineNew); 
		istringstream record(lineNew);
		for (int j = 0; j < 4; ++j) {
			record >> wordNew[j];
			}
		cadence[i] = stoi(wordNew[0]);
		mask[i] = stod(wordNew[1]);
		y[i] = stod(wordNew[2]);
		yerr[i] = stod(wordNew[3]);
		i += 1;
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

		double t_incr = 0.0;
		cout << "Set the sampling interval t_incr such that t_incr > 0.0" << endl;
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
			LnLike = SystemMaster.computeLnLike(numCadences, y, yerr, mask);
			cout << endl;
			}

		cout.precision(16);
		cout << "LnLike: " << fixed << noshowpos << LnLike << endl;
		cout << endl;

		printf("A\n");
		SystemMaster.printA();
		printf("\n");
		printf("B\n");
		SystemMaster.printB();
		printf("\n");
		printf("F\n");
		SystemMaster.printF();
		printf("\n");
		printf("D\n");
		SystemMaster.printD();
		printf("\n");
		printf("Q\n");
		SystemMaster.printQ();
		printf("\n");

		AcquireInput(cout,cin,"Done? (1/0): ","Input not parsed!\n",doneYN);
		}

	SystemMaster.deallocCARMA();
	_mm_free(ThetaMaster);
	_mm_free(y);
	_mm_free(yerr);
	_mm_free(mask);
	}