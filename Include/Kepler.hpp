#ifndef KEPLER_HPP
#define KEPLER_HPP

#include "Constants.hpp"
#include "Spherical.hpp"
#include "Obj.hpp"
//#include "SimData.hpp"

using namespace std;

class KeplerObj: public Obj {
private:
	string ID, Path;
	Equatorial Location;
	int Order, NumQuarters, NumCadences, NumLags, FirstCadence, LastCadence, OverSampleFactor, LengthFactor, NumSimsSI, NumSimsSII, NumSimsCS, MaxEvalsSI, MaxEvalsSII, MaxEvalsCS, NumChiSq, Stage, NumSeeds;
	double StartEpoch, MeanFlux, FracHOST, FracMIC, FTolSI, FTolSII;
	tuple<vector<array<int,2>>,vector<array<double,5>>> readRawData();
	tuple<vector<array<int,2>>,vector<array<double,5>>> readRawData(string& fileName, bool cbvORpdc);
	vector<int> getQuarterList(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray);
	int getQuarterIndex(const vector<int>& quarterList, int quarter);
	vector<int> getQuarterLimits(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray, const vector<int>& quarterList);
	vector<int> getQuarterLengths(const vector<int>& quarterLimits);
	vector<int> getIncrementLengths(const vector<int>& quarterLimits);
	array<int,2> pickMerge(const vector<int>& quarterList, const vector<int>& incrementLengths);
	double getOffset(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray, const vector<int>& quarterList,const vector<int>& quarterLimits, const vector<int>& incrementLengths, array<int,2> mergedQuarters);
	void mergeQuarters(tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray, vector<int>& quarterList, vector<int>& quarterLimits, vector<int>& quarterLengths, vector<int>& incrementLengths, array<int,2> mergedQuarters, int stitchMethod);
public:
	KeplerObj() = default;
	KeplerObj(string id, string path, Equatorial loc);
	string getID();
	string getPath();
	Equatorial getLocation();
	double getRedShift();
	int getOrder();
	void setOrder(int ord);
	int getNumCadences();
	int getNumLags();
	int getOverSampleFactor();
	void setOverSampleFactor(int overSampleFactor);
	int getNumQuarters();
	double getStartEpoch();
	double getMeanFlux();
	int getFirstCadence();
	int getLastCadence();
	void setFracHOST(double fracHOST);
	double getFracHOST();
	void setFracMIC(double fracMIC);
	double getFracMIC();
	int getStage();
	void setStage(int stage);
	void setNumSimsSI(int numSimsSI);
	int getNumSimsSI();
	void setMaxEvalsSI(int maxEvalsSI);
	int getMaxEvalsSI();
	void setFTolSI(double fTolSI);
	double getFTolSI();
	void setNumSimsSII(int numSimsSII);
	int getNumSimsSII();
	void setMaxEvalsSII(int maxEvalsSII);
	int getMaxEvalsSII();
	void setFTolSII(double fTolSII);
	double getFTolSII();
	void setNumSimsCS(int numSimsCS);
	int getNumSimsCS();
	void setMaxEvalsCS();
	int getMaxEvalsCS();
	void setNumChiSq(int numChiSq);
	int getNumChiSq();
	void setNumSeeds(int numSeeds);
	int getNumSeeds();
	tuple<vector<array<int,2>>,vector<array<double,5>>> getData();
	tuple<vector<array<int,2>>,vector<array<double,5>>> getData(int stitchMethod);
	tuple<vector<array<int,2>>,vector<array<double,5>>> getData(bool forceCalibrate, int stitchMethod); // Method0 = No Stitching; Method1 = Match nd points; Method2 = Match Avg across 50 entries; Method3 = Using first increments
	tuple<vector<array<int,2>>,vector<array<double,5>>> getData(string& fileName, bool cbvORpdc);
	tuple<vector<array<int,2>>,vector<array<double,5>>> getData(string& filename, bool cbvORpdc, int stitchMethod);
	tuple<vector<array<int,2>>,vector<array<double,5>>> getData(string& fileName, bool cbvORpdc, bool forceCalibrate, int stitchMethod); // Method0 = No Stitching; Method1 = Match nd points; Method2 = Match Avg across 50 entries; Method3 = Using first increments
	void setProperties(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray);
	void setMask(const tuple<vector<array<int,2>>,vector<array<double,5>>>& dataArray, double* mask);
	vector<unsigned int> getSeeds();
	vector<unsigned int> getSeeds(int numSeedsReq);
	int epochToindex(double epoch);
	double indexToepoch(int index);
	vector<int> epochToindex(vector<double> epochList);
	vector<double> indexToepoch(vector<int> indexList);
	};

bool compareSF1(const array<double,2> &a, const array<double,2> &b);

#endif
