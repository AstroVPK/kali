#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>
#include <boost/system/linux_error.hpp>
#include <boost/filesystem.hpp>
#include <boost/io/detail/quoted_manip.hpp>
#include "Correlation.hpp"
#include "Acquire.hpp"

#define TIME_ACVF
#define TIME_ACF
#define TIME_PACF
#define TIME_SF1

using namespace std;

int main() {
	cout.clear();
	cout << endl;
	cout << "Program: computeCFs" << endl;
	cout << "Purpose: Program to compute the ACF/ACVF and the PACF for light curves" << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institution: Drexel university, Department of Physics" << endl;
	cout << "Email: vpk24@drexel.edu" << endl;
	cout << endl;

	double maxDouble = numeric_limits<double>::max();
	double sqrtMaxDouble = sqrt(maxDouble);

	string basePath;
	vector<string> word(2);

	AcquireDirectory(cout,cin,"Path to working directory: ","Invalid path!\n",basePath);
	basePath += "/";

	string line, yFilePath = basePath + "y.dat";
	cout << "Input LC: " << yFilePath << endl;
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

	cout << "Computing ACVF" << endl;

	double* acvf = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));

	#ifdef TIME_ACVF
	double timeACVFBegin = 0.0, timeACVFEnd = 0.0, timeACVF = 0.0;
	#pragma omp barrier
	timeACVFBegin = dtime();
	#endif

	ACVF(numCadences, y, mask, acvf);

	#ifdef TIME_ACVF
	#pragma omp barrier
	timeACVFEnd = dtime();
	timeACVF = timeACVFEnd - timeACVFBegin;
	cout << "ACVF computed in " << timeACVF << " (s)!" << endl;
	#endif

	cout << "Writing ACVF to ";
	string ACVFPath = basePath + "acvf.dat";
	cout << ACVFPath << endl;
	ofstream ACVFFile;
	ACVFFile.open(ACVFPath);
	ACVFFile.precision(16);
	ACVFFile << noshowpos << fixed << "numCadences: " << numCadences << endl;
	ACVFFile << noshowpos << fixed << "numObservations: " << numObservations << endl;
	ACVFFile << noshowpos << fixed << "numLags: " << numCadences - 1<< endl;
	for (int lagNum = 0; lagNum < numCadences - 1; ++lagNum) {
		ACVFFile << noshowpos << scientific << acvf[lagNum] << endl;
		}
	ACVFFile << noshowpos << scientific << acvf[numCadences - 1];
	ACVFFile.close();
	cout << "ACVF written!" << endl;

	cout << "Computing ACF" << endl;

	double* acf = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));

	#ifdef TIME_ACF
	double timeACFBegin = 0.0, timeACFEnd = 0.0, timeACF = 0.0;
	#pragma omp barrier
	timeACFBegin = dtime();
	#endif

	ACF(numCadences, acvf, acf);

	#ifdef TIME_ACF
	#pragma omp barrier
	timeACFEnd = dtime();
	timeACF = timeACFEnd - timeACFBegin;
	cout << "ACF computed in " << timeACF << " (s)!" << endl;
	#endif

	cout << "Writing ACF to ";
	string ACFPath = basePath + "acf.dat";
	cout << ACFPath << endl;
	ofstream ACFFile;
	ACFFile.open(ACFPath);
	ACFFile.precision(16);
	ACFFile << noshowpos << fixed << "numCadences: " << numCadences << endl;
	ACFFile << noshowpos << fixed << "numObservations: " << numObservations << endl;
	ACFFile << noshowpos << fixed << "numLags: " << numCadences - 1 << endl;
	for (int lagNum = 0; lagNum < numCadences - 1; ++lagNum) {
		ACFFile << noshowpos << scientific << acf[lagNum] << endl;
		}
	ACFFile << noshowpos << scientific << acf[numCadences - 1];
	ACFFile.close();
	cout << "ACF written!" << endl;

	int maxLag = 0;
	AcquireInput(cout,cin,"Maximum number of lags to calculate PACF to: ","Invalid value.\n",maxLag);

	cout << "Computing PACF" << endl;

	double* pacf = static_cast<double*>(_mm_malloc(maxLag*sizeof(double),64));

	#ifdef TIME_PACF
	double timePACFBegin = 0.0, timePACFEnd = 0.0, timePACF = 0.0;
	#pragma omp barrier
	timePACFBegin = dtime();
	#endif

	PACF(numCadences, maxLag, acvf, pacf);

	#ifdef TIME_PACF
	#pragma omp barrier
	timePACFEnd = dtime();
	timePACF = timePACFEnd - timePACFBegin;
	cout << "PACF computed in " << timePACF << " (s)!" << endl;
	#endif

	cout << "Writing PACF to ";
	string PACFPath = basePath + "pacf.dat";
	cout << PACFPath << endl;
	ofstream PACFFile;
	PACFFile.open(PACFPath);
	PACFFile.precision(16);
	PACFFile << noshowpos << fixed << "numCadences: " << numCadences << endl;
	PACFFile << noshowpos << fixed << "numObservations: " << numObservations << endl;
	PACFFile << noshowpos << fixed << "numLags: " << maxLag << endl;
	for (int lagNum = 0; lagNum < maxLag; ++lagNum) {
		PACFFile << noshowpos << scientific << pacf[lagNum] << endl;
		}
	PACFFile << noshowpos << scientific << pacf[maxLag];
	PACFFile.close();
	cout << "PACF written!" << endl;

	cout << "Computing SF1" << endl;

	double* sf1 = static_cast<double*>(_mm_malloc(numCadences*sizeof(double),64));

	#ifdef TIME_SF1
	double timeSF1Begin = 0.0, timeSF1End = 0.0, timeSF1 = 0.0;
	#pragma omp barrier
	timeSF1Begin = dtime();
	#endif

	SF1(numCadences, acvf, sf1);

	#ifdef TIME_SF1
	#pragma omp barrier
	timeSF1End = dtime();
	timeSF1 = timeSF1End - timeSF1Begin;
	cout << "SF1 computed in " << timeSF1 << " (s)!" << endl;
	#endif

	cout << "Writing SF1 to ";
	string SF1Path = basePath + "sf1.dat";
	cout << SF1Path << endl;
	ofstream SF1File;
	SF1File.open(SF1Path);
	SF1File.precision(16);
	SF1File << noshowpos << fixed << "numCadences: " << numCadences << endl;
	SF1File << noshowpos << fixed << "numObservations: " << numObservations << endl;
	SF1File << noshowpos << fixed << "numLags: " << numCadences - 1 << endl;
	for (int lagNum = 0; lagNum < numCadences - 1; ++lagNum) {
		SF1File << noshowpos << scientific << sf1[lagNum] << endl;
		}
	SF1File << noshowpos << scientific << sf1[numCadences - 1];
	SF1File.close();
	cout << "SF1 written!" << endl;

	cout << "Program exiting...Have a nice day!" << endl;

	_mm_free(sf1);
	_mm_free(pacf);
	_mm_free(acf);
	_mm_free(acvf);
	_mm_free(cadence);
	_mm_free(mask);
	_mm_free(y);
	_mm_free(yerr);
	}
