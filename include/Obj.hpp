#ifndef OBJ_HPP
#define OBJ_HPP

#include "Universe.hpp"
#include "Spherical.hpp"

using namespace std;

class Obj {
protected:
	vector<string> Identifier;
	Equatorial Location;
public:
	Obj();
	Obj(array<double,3> pos);
	Obj(Equatorial loc);
	Obj(string identifier, array<double,3> pos);
	Obj(string identifier, Equatorial loc);
	Obj(vector<string> identifier, array<double,3> pos);
	Obj(vector<string> identifier, Equatorial loc);
	virtual ~Obj() = default;
	string getIdentifier(int number);
	vector<string> getIdentifier();
	void addIdentifier(string identifier);
	void addIdentifier(vector<string> identifier);
	Equatorial getLocation();
	double radialComovingDistance(const Universe& u);
	double transverseComovingDistance(const Universe& u);
	double angularDiameterDistance(const Universe& u);
	double luminosityDistance(const Universe& u);
	double lookBackTime(const Universe& u);
	};
	
#endif	
