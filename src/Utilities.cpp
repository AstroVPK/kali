#include <mkl_types.h>
//#include <mathimf.h>
#include <cmath>
#include <vector>
#include <tuple>
#include <exception>
#include <stdexcept>
#include <fstream>
#include "Utilities.hpp"

using namespace std;

/* This function histograms a vector of doubles. All the bins are closed on the bottom and half-open on the top, expect the last
which is closed on both the top and bottom. */
tuple<vector<double>,vector<int>> histogram(vector<double> data, int numBins) {
	vector<int> binCounts(numBins);
	vector<double> binCenters(numBins);
	int dataLen = data.size();
	if (dataLen == 0) {
		tuple<vector<double>,vector<int>> result(binCenters,binCounts);
		return result;
		}
	double dataMax = data[0], dataMin = data[0], binSize;
	for (int i = 0; i < dataLen; i++) {
		if (data[i] > dataMax) {
			dataMax = data[i];
			}
		if (data[i] < dataMin) {
			dataMin = data[i];
			}	
		}  	
	binSize = (dataMax - dataMin)/numBins;
	for (int binCounter = 0; binCounter < numBins-1; binCounter++) {
		double binMin = binCounter*binSize, binMax = (binCounter+1)*binSize, binCent = (binMax + binMin)/2.0;
		int binCount = 0;
		for (int posCounter = 0; posCounter < dataLen; posCounter++) {
			if ((data[posCounter] >= binMin) and (data[posCounter] < binMax)) {
				binCount++;
				}
			}
		binCounts[binCounter] = binCount;
		binCenters[binCounter] = binCent;	
		}
	int binCounter = numBins - 1, binCount = 0;
	double binMin = binCounter*binSize, binMax = (binCounter+1)*binSize, binCent = (binMax + binMin)/2.0;	
	for (int posCounter = 0; posCounter < dataLen; posCounter++) {
		if ((data[posCounter] >= binMin) and (data[posCounter] <= binMax)) {
			binCount++;
			}
		}
	binCounts[binCounter] = binCount;
	binCenters[binCounter] = binCent;
	tuple<vector<double>,vector<int>> result(binCenters,binCounts);
	return result;		
	}
	
tuple<vector<double>,vector<int>> histogram(vector<double> data, int numBins, double base) {
	if ((base > 0.0) and (base != 1.0)) {
		vector<double> plusList, minusList, combBins(2*numBins+1);
		vector<int> combCounts(2*numBins+1);
		tuple<vector<double>,vector<int>> result(combBins,combCounts);
		int numZeros = 0;
		for (vector<double>::iterator iter = data.begin(); iter < data.end(); iter++) {
			if (*iter < 0.0) {
				minusList.push_back(log10(-(*iter))/log10(base));
				}
			if (*iter > 0.0) {
				plusList.push_back(log10(*iter)/log10(base));
				}
			else {
				numZeros = numZeros + 1;
				}	
			}
		tuple<vector<double>,vector<int>> histPlus, histMinus;
		histPlus = histogram(plusList,numBins);
		histMinus = histogram(minusList,numBins);
		for (int combCounter = 0; combCounter < numBins; combCounter++) {
			get<0>(result)[numBins-1-combCounter] = -(pow(base,get<0>(histMinus)[combCounter]));
			get<0>(result)[combCounter+numBins+1] = pow(base,get<0>(histPlus)[combCounter]);
			get<1>(result)[numBins-1-combCounter] = get<1>(histMinus)[combCounter];
			get<1>(result)[combCounter+numBins+1] = get<1>(histPlus)[combCounter];
			}
		get<0>(result)[numBins] = 0.0;
		get<1>(result)[numBins] = numZeros;
		return result;						
		}
	throw runtime_error("Invalid base!");	
	}
	
tuple<vector<double>,vector<int>> histogram(vector<double> data, int numBins, string base) {
	if (base == "e") {
		vector<double> plusList, minusList, combBins(2*numBins+1);
		vector<int> combCounts(2*numBins+1);
		tuple<vector<double>,vector<int>> result(combBins,combCounts);
		int numZeros = 0;
		for (vector<double>::iterator iter = data.begin(); iter < data.end(); iter++) {
			if (*iter < 0.0) {
				minusList.push_back(log(-(*iter)));
				}
			if (*iter > 0.0) {
				plusList.push_back(log(*iter));
				}
			else {
				numZeros = numZeros + 1;
				}	
			}
		tuple<vector<double>,vector<int>> histPlus, histMinus;
		histPlus = histogram(plusList,numBins);
		histMinus = histogram(minusList,numBins);
		for (int combCounter = 0; combCounter < numBins; combCounter++) {
			get<0>(result)[numBins-1-combCounter] = -(exp(get<0>(histMinus)[combCounter]));
			get<0>(result)[combCounter+numBins+1] = exp(get<0>(histPlus)[combCounter]);
			get<1>(result)[numBins-1-combCounter] = get<1>(histMinus)[combCounter];
			get<1>(result)[combCounter+numBins+1] = get<1>(histPlus)[combCounter];
			}
		get<0>(result)[numBins] = 0.0;
		get<1>(result)[numBins] = numZeros;
		return result;						
		}
	throw runtime_error("Unrecognized base!");	
	}
	
bool exists(const string& filename)
{
  ifstream ifile;
  ifile.open(filename);
  return ifile;
}				

int gcd(int a, int b) {
	int c;
	while (a != 0) {
		c = a; a = b%a;  b = c;
		}
	return b;
	}

int lcm(int a, int b) {
	return a*b/gcd(a,b);
	}