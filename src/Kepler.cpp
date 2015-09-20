#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <mkl.h>
#include <mkl_types.h>
#include <nlopt.hpp>
//#include "SimData.hpp"
#include "Kepler.hpp"
//#include "ComputeSim.hpp"
//#include "PRH.hpp"
#include "Utilities.hpp"

using namespace std;
using namespace nlopt;

KeplerObj::KeplerObj(string id, string path, Equatorial loc) {
	string dataListFile, calibratedDataFile;
	dataListFile=path+id+"/"+id+"-epochList.dat";
	calibratedDataFile=path+id+"/"+id+"-calibrated.dat";
	ifstream dataList(dataListFile); ifstream calibratedData(calibratedDataFile);
	if ((dataList) or (calibratedData)) {
		Path = path;
		ID = id;
		Order = 0;
		Location = loc;
		StartEpoch = 0.0;
		FirstCadence = 0;
		LastCadence = 0;
		MeanFlux = 0.0;
		NumCadences = 0;
		NumLags = 0;
		OverSampleFactor = 10;
		LengthFactor = 10;
		NumSimsSI = 100;
		MaxEvalsSI = 1;
		FTolSI=0.05;
		NumSimsSII = 1000;
		MaxEvalsSII = 1;
		FTolSI=0.01;
		NumSimsCS = 10000;
		MaxEvalsCS = 1;
		NumQuarters = 0;
		}
	else {
		cout << dataListFile << endl;
		cout << calibratedDataFile << endl;
		cout << "List of input raw data files or calibrated data file not found. Could not instantiate object." << endl;
		}
	}

string KeplerObj::getID() {
	return ID;
	}

string KeplerObj::getPath() {
	return Path;
	}

Equatorial KeplerObj::getLocation() {
	return Location;
	}
	
double KeplerObj::getRedShift() {
	return Location.getRedShift();
	}

int KeplerObj::getOrder() {
	return Order;
	}

void KeplerObj::setOrder(int ord) {
	Order=ord;
	}

int KeplerObj::getNumCadences() {
	return NumCadences;
	}

int KeplerObj::getNumLags() {
	return NumLags;
	}

int KeplerObj::getOverSampleFactor() {
	return OverSampleFactor;
	}

void KeplerObj::setOverSampleFactor(int overSampleFactor) {
	OverSampleFactor = overSampleFactor;
	}

double KeplerObj::getFracHOST() {
	return FracHOST;
	}

void KeplerObj::setFracHOST(double fracHOST) {
	FracHOST = fracHOST;
	}

double KeplerObj::getFracMIC() {
	return FracMIC;
	}

void KeplerObj::setFracMIC(double fracMIC) {
	FracMIC = fracMIC;
	}

int KeplerObj::getStage() {
	return Stage;
	}

void KeplerObj::setStage(int stage) {
	Stage = stage;
	}

int KeplerObj::getNumSimsSI() {
	return NumSimsSI;
	}

void KeplerObj::setNumSimsSI(int numSimsSI) {
	NumSimsSI = numSimsSI;
	}

void KeplerObj::setMaxEvalsSI(int maxEvalsSI) {
	MaxEvalsSI = maxEvalsSI;
	}

int KeplerObj::getMaxEvalsSI() {
	return MaxEvalsSI;
	}

void KeplerObj::setFTolSI(double fTolSI) {
	FTolSI = fTolSI;
	}

double KeplerObj::getFTolSI() {
	return FTolSI;
	}

int KeplerObj::getNumSimsSII() {
	return NumSimsSII;
	}

void KeplerObj::setNumSimsSII(int numSimsSII) {
	NumSimsSII = numSimsSII;
	}

void KeplerObj::setMaxEvalsSII(int maxEvalsSII) {
	MaxEvalsSII = maxEvalsSII;
	}

int KeplerObj::getMaxEvalsSII() {
	return MaxEvalsSII;
	}

void KeplerObj::setFTolSII(double fTolSII) {
	FTolSII = fTolSII;
	}

double KeplerObj::getFTolSII() {
	return FTolSII;
	}

int KeplerObj::getNumSimsCS() {
	return NumSimsCS;
	}

void KeplerObj::setNumSimsCS(int numSimsCS) {
	NumSimsCS = numSimsCS;
	}

void KeplerObj::setMaxEvalsCS() {
	MaxEvalsCS = 1;
	}
	
int KeplerObj::getMaxEvalsCS() {
	return MaxEvalsCS;
	}

int KeplerObj::getNumChiSq() {
	return NumChiSq;
	}

void KeplerObj::setNumChiSq(int numChiSq) {
	NumChiSq = numChiSq;
	}

int KeplerObj::getNumSeeds() {
	return NumSeeds;
	}

void KeplerObj::setNumSeeds(int numSeeds) {
	NumSeeds = numSeeds;
	}

int KeplerObj::getNumQuarters() {
	return NumQuarters;
	}

double KeplerObj::getMeanFlux() {
	return MeanFlux;
	}

double KeplerObj::getStartEpoch() {
	return StartEpoch;
	}

tuple<vector<array<int,2>>,vector<array<double,5>>> KeplerObj::readRawData() {
	vector<double> time, time_err, pdcsap_flux, pdcsap_flux_err;
	vector<int> cadenceNo, quarterNo;
	double dummy = 0.0;
	string listFile = Path+ID+"/"+ID+"-epochList.dat";
	ifstream fileList;
	fileList.open(listFile);
	string dataFileName, dataFilePath, line;
	int quarterCounter = 0, numEntries = 0;
	while (!fileList.eof()) {
		getline(fileList,dataFileName);
		dataFilePath = Path+ID+"/"+dataFileName;
		ifstream dataFile;
		dataFile.open(dataFilePath);
		getline(dataFile,line);
		quarterCounter = ++quarterCounter;
		while (!dataFile.eof()) {
			getline(dataFile,line);
			istringstream record(line);
			vector<string> word(20);
			for (vector<string>::iterator wordIter = word.begin(); wordIter != word.end(); ++wordIter) {
				record >> *wordIter;
				}
			if ((istringstream(word[7]) >> dummy) and (stoi(word[9]) == 0)) {
				time.push_back(stod(word[0]));
				time_err.push_back(stod(word[1]));
				cadenceNo.push_back((int)stoi(word[2]));
				pdcsap_flux.push_back(stod(word[7]));
				pdcsap_flux_err.push_back(stod(word[8])); 
				quarterNo.push_back(quarterCounter);
				NumQuarters = quarterCounter;
				numEntries += 1;
				}
			word.clear();
			}
		dataFile.close();
		}
	fileList.close();
	vector<int>::iterator cadenceBegin = cadenceNo.begin(), cadenceEnd = cadenceNo.end();
	int usedCounter = 0, countCadences = 0;
	double redShift = Location.getRedShift();
	array<int,2> cadenceTempArr;
	array<double,5> dataTempArr;
	vector<array<int,2>> cadenceTempVec;
	vector<array<double,5>> dataTempVec;
	for (int cadenceCounter = *(cadenceBegin); cadenceCounter < *(cadenceEnd-1)+1; cadenceCounter++) {
		countCadences += 1;
		cadenceTempArr[0] = cadenceCounter;
		dataTempArr[0] = (double)cadenceCounter*lcCadence/((double)1.0+redShift);
		if (cadenceNo[usedCounter] == cadenceCounter) {
			cadenceTempArr[1] = quarterNo[usedCounter];
			dataTempArr[1] = pdcsap_flux[usedCounter];
			dataTempArr[2] = pdcsap_flux_err[usedCounter];
			dataTempArr[3] = time[usedCounter];
			dataTempArr[4] = time_err[usedCounter];
			usedCounter += 1;
			}
		else {
			cadenceTempArr[1] = (int)-1;
			dataTempArr[0] = (double)0.0;
			dataTempArr[1] = (double)0.0;
			dataTempArr[2] = (double)0.0;
			dataTempArr[3] = (double)0.0;
			dataTempArr[4] = (double)0.0;
			}
		cadenceTempVec.push_back(cadenceTempArr);
		dataTempVec.push_back(dataTempArr);
		}
	NumCadences = countCadences;
	NumLags = NumCadences - 1;
	int cadenceTempVecSize = cadenceTempVec.size();
	cadenceTempVec.resize(cadenceTempVecSize);
	int dataTempVecSize = dataTempVec.size();
	dataTempVec.resize(dataTempVecSize);
	tuple<vector<array<int,2>>,vector<array<double,5>>> resultArr(cadenceTempVec,dataTempVec);
	cadenceTempVec.clear();
	dataTempVec.clear();
	return resultArr;
	}

tuple<vector<array<int,2>>,vector<array<double,5>>> KeplerObj::readRawData(string& fileName, bool cbvORpdc) {
	int colNum_FLUX = 7, colNum_FLUX_ERR = 8;
	if (cbvORpdc == 0) { // Use PDC
		colNum_FLUX = 7;
		colNum_FLUX_ERR = 8;
		} else {
		colNum_FLUX = 21;
		colNum_FLUX_ERR = 8;
		}
	vector<double> time, time_err, pdcsap_flux, pdcsap_flux_err;
	vector<int> cadenceNo, quarterNo;
	double dummy = 0.0;
	string listFile = Path+ID+"/"+fileName;
	ifstream fileList;
	fileList.open(listFile);
	string dataFileName, dataFilePath, line;
	int quarterCounter = 0, numEntries = 0;
	while (!fileList.eof()) {
		getline(fileList,dataFileName);
		dataFilePath = Path+ID+"/"+dataFileName;
		ifstream dataFile;
		dataFile.open(dataFilePath);
		getline(dataFile,line);
		quarterCounter = ++quarterCounter;
		while (!dataFile.eof()) {
			getline(dataFile,line);
			istringstream record(line);
			vector<string> word(50);
			for (vector<string>::iterator wordIter = word.begin(); wordIter != word.end(); ++wordIter) {
				record >> *wordIter;
				}
			if ((istringstream(word[7]) >> dummy) and (stoi(word[9]) == 0)) {
				time.push_back(stod(word[0]));
				time_err.push_back(stod(word[1]));
				cadenceNo.push_back((int)stoi(word[2]));
				pdcsap_flux.push_back(stod(word[colNum_FLUX]));
				pdcsap_flux_err.push_back(stod(word[colNum_FLUX_ERR]));
				quarterNo.push_back(quarterCounter);
				NumQuarters = quarterCounter;
				numEntries += 1;
				}
			word.clear();
			}
		dataFile.close();
		}
	fileList.close();
	vector<int>::iterator cadenceBegin = cadenceNo.begin(), cadenceEnd = cadenceNo.end();
	int usedCounter = 0, countCadences = 0;
	double redShift = Location.getRedShift();
	array<int,2> cadenceTempArr;
	array<double,5> dataTempArr;
	vector<array<int,2>> cadenceTempVec;
	vector<array<double,5>> dataTempVec;
	for (int cadenceCounter = *(cadenceBegin); cadenceCounter < *(cadenceEnd-1)+1; cadenceCounter++) {
		countCadences += 1;
		cadenceTempArr[0] = cadenceCounter;
		dataTempArr[0] = (double)cadenceCounter*lcCadence/((double)1.0+redShift);
		if (cadenceNo[usedCounter] == cadenceCounter) {
			cadenceTempArr[1] = quarterNo[usedCounter];
			dataTempArr[1] = pdcsap_flux[usedCounter];
			dataTempArr[2] = pdcsap_flux_err[usedCounter];
			dataTempArr[3] = time[usedCounter];
			dataTempArr[4] = time_err[usedCounter];
			usedCounter += 1;
			}
		else {
			cadenceTempArr[1] = (int)-1;
			dataTempArr[0] = (double)0.0;
			dataTempArr[1] = (double)0.0;
			dataTempArr[2] = (double)0.0;
			dataTempArr[3] = (double)0.0;
			dataTempArr[4] = (double)0.0;
			}
		cadenceTempVec.push_back(cadenceTempArr);
		dataTempVec.push_back(dataTempArr);
		}
	NumCadences = countCadences;
	NumLags = NumCadences - 1;
	int cadenceTempVecSize = cadenceTempVec.size();
	cadenceTempVec.resize(cadenceTempVecSize);
	int dataTempVecSize = dataTempVec.size();
	dataTempVec.resize(dataTempVecSize);
	tuple<vector<array<int,2>>,vector<array<double,5>>> resultArr(cadenceTempVec,dataTempVec);
	cadenceTempVec.clear();
	dataTempVec.clear();
	return resultArr;
	}

vector<int> KeplerObj::getQuarterList(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray) {
	vector<int> quarterList;
	int quarter, setContinue;
	for (int cadenceCounter = 1; cadenceCounter < NumCadences; cadenceCounter++) {
		setContinue = 0;
		quarter = get<0>(dataArray)[cadenceCounter][1];
		if (quarter > 0) {
			for (int quarterCounter = 0; quarterCounter < quarterList.size(); quarterCounter++) {
				if (quarterList[quarterCounter] == quarter) {
					setContinue = 1;
					break;
					}
				}
			if (setContinue == 1) {
				continue;
				}
			else {
				quarterList.push_back(quarter);
				}
			}
		else {
			continue;
			}
		}
	return quarterList;
	}

int KeplerObj::getQuarterIndex(const vector<int>& quarterList, int quarter) {
	int quarterIndex = 0;
	while (quarterList[quarterIndex] < quarter) {
		quarterIndex++;
		}
	return quarterIndex;
	}

vector<int> KeplerObj::getQuarterLimits(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray, const vector<int>& quarterList) {
	vector<int> quarterLimits;
	vector<array<int,2>> cadenceVec = get<0>(dataArray);
	vector<array<double,5>> dataVec = get<1>(dataArray);
	vector<array<int,2>>::iterator cadBegin = cadenceVec.begin(), cadEnd = cadenceVec.end();
	array<int,2> initialCad = *(cadBegin), finalCad = *(cadEnd-1), currentCad;
	array<double,5> currentData;
	int quartCadLow, quartCadHigh;
	for (int quartCounter = 0; quartCounter < quarterList.size(); quartCounter++) {
		quartCadLow = 1000000, quartCadHigh = 0;
		for (int cadCounter = 0; cadCounter < NumCadences; cadCounter++) {
			currentCad = cadenceVec[cadCounter];
			currentData = dataVec[cadCounter];
			if (currentCad[1] == quarterList[quartCounter]) {
				if ((currentCad[0] < quartCadLow) and (currentData[1] > 0.0)) {
					quartCadLow = currentCad[0];
					}
				if ((currentCad[0] > quartCadHigh) and (currentData[1] > 0.0)) {
					quartCadHigh = currentCad[0];
					}
				}
			}
		quarterLimits.push_back(quartCadLow);
		quarterLimits.push_back(quartCadHigh);	
		}
	return quarterLimits;
	}

vector<int> KeplerObj::getQuarterLengths(const vector<int>& quarterLimits) {
	int lenQuarterLimits = quarterLimits.size();
	vector<int> quarterLengths(lenQuarterLimits/2);
	for (int quartCounter = 0; quartCounter < (lenQuarterLimits/2); quartCounter++) {
		quarterLengths[quartCounter] = quarterLimits[2*quartCounter+1]-quarterLimits[2*quartCounter];
		}
	return quarterLengths;
	}

vector<int> KeplerObj::getIncrementLengths(const vector<int>& quarterLimits) {
	int lenQuarterLimits = quarterLimits.size();
	vector<int> incrementLengths((lenQuarterLimits/2)-1);
	for (int quartCounter = 0; quartCounter < ((lenQuarterLimits/2)-1); quartCounter++) {
		incrementLengths[quartCounter] = quarterLimits[2*(quartCounter+1)]-quarterLimits[2*quartCounter+1];
		}
	return incrementLengths;
	}

array<int,2> KeplerObj::pickMerge(const vector<int>& quarterList, const vector<int>& incrementLengths) {
	int lowerQuarter, upperQuarter, minVal;
	if (incrementLengths.size() == 0) {
		lowerQuarter = 0;
		}
	else {
		minVal = *(min_element(incrementLengths.begin(), incrementLengths.end()));
		for (int incrCounter = 0; incrCounter < incrementLengths.size(); incrCounter++) {
			if (incrementLengths[incrCounter] == minVal) {
				lowerQuarter = quarterList[incrCounter];
				upperQuarter = quarterList[incrCounter+1];
				}
			}
		}
	array<int,2> mergedQuarters = {lowerQuarter,upperQuarter};
	return mergedQuarters;
	}

double KeplerObj::getOffset(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray, const vector<int>& quarterList, const vector<int>& quarterLimits, const vector<int>& incrementLengths, array<int,2> mergedQuarters) {
	int sumCounter = 0, quarterIndex, lowerQuarterIndex = getQuarterIndex(quarterList, mergedQuarters[0]), upperQuarterIndex = getQuarterIndex(quarterList, mergedQuarters[1]), jump = incrementLengths[lowerQuarterIndex];
	double sum = 0.0, firstIncrement = 0.0, correction = 0.0;	
	for (int cadCounter = 0; cadCounter < (NumCadences-jump); cadCounter++) {
		if ((get<1>(dataArray)[cadCounter][1] > 0.0) and (get<1>(dataArray)[cadCounter+jump][1] > 0.0) and (get<0>(dataArray)[cadCounter][1] == get<0>(dataArray)[cadCounter+jump][1])) {
			sum += (get<1>(dataArray)[cadCounter+jump][1] - get<1>(dataArray)[cadCounter][1]);
			sumCounter += 1;
			}
		}
	if (sumCounter > 0) {
		firstIncrement = sum/sumCounter;
		}
	correction = (get<1>(dataArray)[quarterLimits[2*upperQuarterIndex]-quarterLimits[0]][1])/(get<1>(dataArray)[quarterLimits[2*lowerQuarterIndex+1]-quarterLimits[0]][1] + firstIncrement);
	return correction;
	}

void KeplerObj::mergeQuarters(tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray, vector<int>& quarterList, vector<int>& quarterLimits, vector<int>& quarterLengths, vector<int>& incrementLengths, array<int,2> mergedQuarters, int stitchMethod) {
	double correction = 1.0;
	if (stitchMethod == 0) {
		// TEST THIS!!!! Method0 leaves the curve uncalibrated!!!!
		correction = 1.0;
		}
	else if (stitchMethod == 1) {
		// TEST THIS!!! Method1 just matches endpoints!
		int sumCounter = 0, quarterIndex, lowerQuarterIndex = getQuarterIndex(quarterList, mergedQuarters[0]), upperQuarterIndex = getQuarterIndex(quarterList, mergedQuarters[1]), jump = incrementLengths[lowerQuarterIndex];
		correction = (get<1>(dataArray)[quarterLimits[2*upperQuarterIndex]-quarterLimits[0]][1])/(get<1>(dataArray)[quarterLimits[2*lowerQuarterIndex+1]-quarterLimits[0]][1]);
		} 
	else if (stitchMethod == 2) {
		// TEST THIS!!! Method2 matches the average of the 50 points before and after the inter-quarter gap
		int numPts = 96;
		int lowerQuarterIndex = getQuarterIndex(quarterList, mergedQuarters[0]), upperQuarterIndex = getQuarterIndex(quarterList, mergedQuarters[1]), jump = incrementLengths[lowerQuarterIndex];
		
		int startPoint = quarterLimits[2*upperQuarterIndex]-quarterLimits[0], endPoint =  quarterLimits[2*lowerQuarterIndex+1]-quarterLimits[0];
		
		double avgQ1 = 0.0, avgQ2 = 0.0;
		int pCounter1 = 0, pCounter2 = 0;
		for (int counter = 0; counter < numPts; counter++) {
			if (get<1>(dataArray)[endPoint-counter][1] > 0.0) {
				avgQ1 += get<1>(dataArray)[endPoint-counter][1];
				pCounter1 += 1;
				}
			if (get<1>(dataArray)[startPoint+counter][1] > 0.0) {
				avgQ2 += get<1>(dataArray)[startPoint+counter][1];
				pCounter2 += 1;
				}
			}
		avgQ1 /= static_cast<double>(pCounter1);
		avgQ2 /= static_cast<double>(pCounter2);
		correction = avgQ2/avgQ1;
		}
	else {
		// Method3 matches the expectation value of the first increment. WORKS!!!!!!
		correction = getOffset(dataArray, quarterList, quarterLimits, incrementLengths, mergedQuarters);
		}
	// Now that we have determined the value of 'correction', we apply it to the light curve!!!!!	
	int quarterIndexLower = getQuarterIndex(quarterList, mergedQuarters[0]), quarterIndexUpper = getQuarterIndex(quarterList, mergedQuarters[1]);
	for (int cadCounter = 0; cadCounter < NumCadences; cadCounter++) {
		if (get<0>(dataArray)[cadCounter][1] > mergedQuarters[0]) {
			if (get<0>(dataArray)[cadCounter][1] == mergedQuarters[1]) {
				get<0>(dataArray)[cadCounter][1] = mergedQuarters[0];
				}
			if (get<1>(dataArray)[cadCounter][1] > 0.0) {
				get<1>(dataArray)[cadCounter][1] /= correction;
				}
			}
		}
	vector<int> newQuarterList, newQuarterLimits, newQuarterLengths, newIncrementLengths;
	vector<int>::iterator posErase1, posErase2, dump;
	newQuarterList = getQuarterList(dataArray);
	quarterList.swap(newQuarterList);
	newQuarterLimits = getQuarterLimits(dataArray, quarterList);
	quarterLimits.swap(newQuarterLimits);
	newQuarterLengths = getQuarterLengths(quarterLimits);
	quarterLengths.swap(newQuarterLengths);
	newIncrementLengths = getIncrementLengths(quarterLimits);
	incrementLengths.swap(newIncrementLengths);
	}

tuple<vector<array<int,2>>,vector<array<double,5>>> KeplerObj::getData() {
	bool forceCalibrate = false;
	tuple<vector<array<int,2>>,vector<array<double,5>>> dataArray = getData(forceCalibrate, 2);
	return dataArray;
	}

tuple<vector<array<int,2>>,vector<array<double,5>>> KeplerObj::getData(int stitchMethod) {
	bool forceCalibrate = true;
	tuple<vector<array<int,2>>,vector<array<double,5>>> dataArray = getData(forceCalibrate, stitchMethod);
	return dataArray;
	}

tuple<vector<array<int,2>>,vector<array<double,5>>> KeplerObj::getData(bool forceCalibrate, int stitchMethod) {
	double dummy;
	array<int,2> cadenceArr;
	array<double,5> dataArr;
	vector<array<int,2>> cadenceVec;
	vector<array<double,5>> dataVec;
	string dataFilePath = Path + ID + "/" + ID + "-calibrated.dat";
	bool calibratedDataExists = exists(dataFilePath);
	if ((forceCalibrate == true) or (calibratedDataExists == false)) {
		//Calibration to be redone or no calibration exists at the moment.
		//Perform calibration by calling the calibration functions. Write the result out to file. Return calibrated dataArray.
		array<int,2> mergedQuarters;
		tuple<vector<array<int,2>>,vector<array<double,5>>> dataArray = readRawData();
		vector<int> quarterList = getQuarterList(dataArray);
		vector<int> quarterLimits = getQuarterLimits(dataArray, quarterList);
		vector<int> quarterLengths = getQuarterLengths(quarterLimits);
		vector<int> incrementLengths = getIncrementLengths(quarterLimits);
		while (quarterLimits.size() != 2) {
			mergedQuarters = pickMerge(quarterList, incrementLengths);
			mergeQuarters(dataArray, quarterList, quarterLimits, quarterLengths, incrementLengths, mergedQuarters, stitchMethod);
			}
		// Write out data.
		ofstream calibratedDataFile;
		calibratedDataFile.open(dataFilePath);
		calibratedDataFile << "NumCadences: " << NumCadences << endl; 
		calibratedDataFile << "cadence" << " quarter" << " cal_time" << " pdcsap_flux" << " pdcsap_flux_err" << " time" << " time_err" << endl;
		calibratedDataFile.precision(16);
		{
			vector<array<int,2>>::iterator cadIter;
			vector<array<double,5>>::iterator dataIter;
			for (cadIter = get<0>(dataArray).begin(), dataIter = get<1>(dataArray).begin(); cadIter < get<0>(dataArray).end()-1, dataIter < get<1>(dataArray).end()-1; cadIter++, dataIter++) {
				//CHANGED!
				calibratedDataFile << noshowpos << scientific << (*cadIter)[0] << " " << (*cadIter)[1] << " " << ((*dataIter)[0]/secPerSiderealDay) << " " << (*dataIter)[1] << " " << (*dataIter)[2] << " " << (*dataIter)[3] << " " << (*dataIter)[4] << endl;
				}
			calibratedDataFile << noshowpos << scientific << (*cadIter)[0] << " " << (*cadIter)[1] << " " << ((*dataIter)[0]/secPerSiderealDay) << " " << (*dataIter)[1] << " " << (*dataIter)[2] << " " << (*dataIter)[3] << " " << (*dataIter)[4];		
			}
		calibratedDataFile.close();
		FirstCadence = get<0>(dataArray)[0][0];
		LastCadence = get<0>(dataArray)[NumCadences-1][0];
		return dataArray;
		}
	else {
		ifstream dataFile;
		dataFile.open(dataFilePath);
		string line;
		getline(dataFile,line);
		getline(dataFile,line);
		int cadCounter = 0;
		while (!dataFile.eof()) {
			getline(dataFile,line);
			cadCounter++;
			istringstream record(line);
			vector<string> word(7);
			for (vector<string>::iterator wordIter = word.begin(); wordIter != word.end(); ++wordIter) {
				record >> *wordIter;
				}
			cadenceArr[0] = (int)stoi(word[0]);
			cadenceArr[1] = (int)stoi(word[1]);
			dataArr[0] = stod(word[2]);
			dataArr[1] = stod(word[3]);
			dataArr[2] = stod(word[4]);
			dataArr[3] = stod(word[5]);
			dataArr[4] = stod(word[6]);
			word.clear();
			cadenceVec.push_back(cadenceArr);
			dataVec.push_back(dataArr);
			}
		NumCadences = cadCounter;
		NumLags = cadCounter-1;
		NumQuarters = 1;
		dataFile.close();
		tuple<vector<array<int,2>>,vector<array<double,5>>> dataArray(cadenceVec,dataVec);
		FirstCadence = get<0>(dataArray)[0][0];
		LastCadence = get<0>(dataArray)[NumCadences-1][0];
		return dataArray;
		}
	}

tuple<vector<array<int,2>>,vector<array<double,5>>> KeplerObj::getData(string& fileName, bool cbvORpdc, bool forceCalibrate, int stitchMethod) {
	double dummy;
	array<int,2> cadenceArr;
	array<double,5> dataArr;
	vector<array<int,2>> cadenceVec;
	vector<array<double,5>> dataVec;
	string dataFilePath = Path + ID + "/" + ID + "-calibrated.dat";
	bool calibratedDataExists = exists(dataFilePath);
	if ((forceCalibrate == true) or (calibratedDataExists == false)) {
		//Calibration to be redone or no calibration exists at the moment.
		//Perform calibration by calling the calibration functions. Write the result out to file. Return calibrated dataArray.
		array<int,2> mergedQuarters;
		tuple<vector<array<int,2>>,vector<array<double,5>>> dataArray = readRawData(fileName, cbvORpdc);
		vector<int> quarterList = getQuarterList(dataArray);
		vector<int> quarterLimits = getQuarterLimits(dataArray, quarterList);
		vector<int> quarterLengths = getQuarterLengths(quarterLimits);
		vector<int> incrementLengths = getIncrementLengths(quarterLimits);
		while (quarterLimits.size() != 2) {
			mergedQuarters = pickMerge(quarterList, incrementLengths);
			mergeQuarters(dataArray, quarterList, quarterLimits, quarterLengths, incrementLengths, mergedQuarters, stitchMethod);
			}
		// Write out data.
		ofstream calibratedDataFile;
		calibratedDataFile.open(dataFilePath);
		calibratedDataFile << "NumCadences: " << NumCadences << endl; 
		calibratedDataFile << "cadence" << " quarter" << " cal_time" << " flux" << " flux_err" << " time" << " time_err" << endl;
		calibratedDataFile.precision(16);
		{
			vector<array<int,2>>::iterator cadIter;
			vector<array<double,5>>::iterator dataIter;
			for (cadIter = get<0>(dataArray).begin(), dataIter = get<1>(dataArray).begin(); cadIter < get<0>(dataArray).end()-1, dataIter < get<1>(dataArray).end()-1; cadIter++, dataIter++) {
				//CHANGED!
				calibratedDataFile << noshowpos << scientific << (*cadIter)[0] << " " << (*cadIter)[1] << " " << ((*dataIter)[0]/secPerSiderealDay) << " " << (*dataIter)[1] << " " << (*dataIter)[2] << " " << (*dataIter)[3] << " " << (*dataIter)[4] << endl;
				}
			calibratedDataFile << noshowpos << scientific << (*cadIter)[0] << " " << (*cadIter)[1] << " " << ((*dataIter)[0]/secPerSiderealDay) << " " << (*dataIter)[1] << " " << (*dataIter)[2] << " " << (*dataIter)[3] << " " << (*dataIter)[4];		
			}
		calibratedDataFile.close();
		FirstCadence = get<0>(dataArray)[0][0];
		LastCadence = get<0>(dataArray)[NumCadences-1][0];
		return dataArray;
		}
	else {
		ifstream dataFile;
		dataFile.open(dataFilePath);
		string line;
		getline(dataFile,line);
		getline(dataFile,line);
		int cadCounter = 0;
		while (!dataFile.eof()) {
			getline(dataFile,line);
			cadCounter++;
			istringstream record(line);
			vector<string> word(7);
			for (vector<string>::iterator wordIter = word.begin(); wordIter != word.end(); ++wordIter) {
				record >> *wordIter;
				}
			cadenceArr[0] = (int)stoi(word[0]);
			cadenceArr[1] = (int)stoi(word[1]);
			dataArr[0] = stod(word[2]);
			dataArr[1] = stod(word[3]);
			dataArr[2] = stod(word[4]);
			dataArr[3] = stod(word[5]);
			dataArr[4] = stod(word[6]);
			word.clear();
			cadenceVec.push_back(cadenceArr);
			dataVec.push_back(dataArr);
			}
		NumCadences = cadCounter;
		NumLags = cadCounter-1;
		NumQuarters = 1;
		dataFile.close();
		tuple<vector<array<int,2>>,vector<array<double,5>>> dataArray(cadenceVec,dataVec);
		FirstCadence = get<0>(dataArray)[0][0];
		LastCadence = get<0>(dataArray)[NumCadences-1][0];
		return dataArray;
		}
	}

tuple<vector<array<int,2>>,vector<array<double,5>>> KeplerObj::getData(string& fileName, bool cbvORpdc) {
	bool forceCalibrate = false;
	tuple<vector<array<int,2>>,vector<array<double,5>>> dataArray = getData(fileName, cbvORpdc, forceCalibrate, 2);
	return dataArray;
	}

tuple<vector<array<int,2>>,vector<array<double,5>>> KeplerObj::getData(string& fileName, bool cbvORpdc, int stitchMethod) {
	bool forceCalibrate = true;
	tuple<vector<array<int,2>>,vector<array<double,5>>> dataArray = getData(fileName, cbvORpdc, forceCalibrate, stitchMethod);
	return dataArray;
	}

int KeplerObj::getFirstCadence() {
	return FirstCadence;
	}

int KeplerObj::getLastCadence() {
	return LastCadence;
	}

void KeplerObj::setProperties(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray) {
	StartEpoch = (get<1>(dataArray)[0][0])/secPerSiderealDay;
	int pointCounter = 0;
	for (int i = 0; i < NumCadences; i++) {
		if (get<1>(dataArray)[i][1] > 0.0) {
			MeanFlux += get<1>(dataArray)[i][1];
			pointCounter++;
			}
		}
	MeanFlux /= pointCounter;
	}

void KeplerObj::setMask(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray, double* mask) {
	for (int i = 0; i < NumCadences; i++) {
		if (get<1>(dataArray)[i][1] > (double)0.0) {
			mask[i] = (double)1.0;
			}
		else {
			mask[i] = (double)0.0;
			}
		}
	}

vector<unsigned int> KeplerObj::getSeeds() {
	string seedsFilePath = Path + ID + "/sim/" + ID + "-seeds.dat";
	bool seedsExist = exists(seedsFilePath);
	string line, word;
	vector<unsigned int> seedsVector(0);
	if (seedsExist) {
		ifstream seedsFile;
		seedsFile.open(seedsFilePath);
		getline(seedsFile,line);
		istringstream record(line);
		record >> word; record >> word;
		int numSeeds = stoi(word);
		seedsVector.resize(numSeeds);
		for (int i = 0; i < numSeeds; i++) {
			getline(seedsFile,line);
			istringstream record(line);
			record >> word;
			seedsVector[i] = static_cast<unsigned int>(stoul(word));
			}
		seedsFile.close();
		}
	return seedsVector;
	}

vector<unsigned int> KeplerObj::getSeeds(int numSeedsReq) {
	string seedsFilePath = Path + ID + "/sim/" + ID + "-seeds.dat";
	bool seedsExist = exists(seedsFilePath);
	string line, word;
	vector<unsigned int> seedsVector(0);
	if (seedsExist) {
		ifstream seedsFile;
		seedsFile.open(seedsFilePath);
		getline(seedsFile,line);
		istringstream record(line);
		record >> word; record >> word;
		int numSeeds = stoi(word);
		if (numSeedsReq > numSeeds) {
			cout << "Not enough seeds in file! - Write more seeds." << endl;
			exit(EXIT_FAILURE);
			}
		seedsVector.resize(numSeedsReq);
		for (int i = 0; i < numSeedsReq; i++) {
			getline(seedsFile,line);
			istringstream record(line);
			record >> word;
			seedsVector[i] = static_cast<unsigned int>(stoul(word));
			}
		seedsFile.close();
		}
	return seedsVector;
	}

int KeplerObj::epochToindex(double epoch) {
	int result = (int)(epoch/(lcCadence/(1.0+Location.getRedShift())));
	return result;
	}

double KeplerObj::indexToepoch(int index) {
	double result = (double)(index*(lcCadence/(1.0+Location.getRedShift())));
	return result;
	}

vector<int> KeplerObj::epochToindex(vector<double> epochList) {
	int epochLen = epochList.size();
	vector<int> result(epochLen);
	double redShift = Location.getRedShift();
	for (int i = 0; i < epochLen; i++) {
		result[i] = (int)(epochList[i]/(lcCadence/(1.0+redShift)));
		}
	return result;
	}

vector<double> KeplerObj::indexToepoch(vector<int> indexList) {
	int indexLen = indexList.size();
	vector<double> result(indexLen);
	double redShift = Location.getRedShift();
	for (int i = 0; i < indexLen; i++) {
		result[i] = (double)(indexList[i]*(lcCadence/(1.0+redShift)));
		}
	return result;
	}	

bool compareSF1(const array<double,2> &a, const array<double,2> &b) {
	bool result = false;
	if (a[1] < b[1]) {
		result = true;
		}
	return result;
	}
