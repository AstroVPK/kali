#include <string>
#include <vector>
#include <array>
#include <stdexcept>
#include "Spherical.hpp"
#include "Obj.hpp"

using namespace std;

Obj::Obj() {
	Identifier.push_back("");
	Location.Position[0] = 0.0;
	Location.Position[1] = 0.0;
	Location.Position[2] = 0.0;
	}

Obj::Obj(array<double,3> pos) {
	Identifier.push_back("");
	Location.Position = pos;
	}
	
Obj::Obj(string identifier, array<double,3> pos) {
	Identifier.push_back(identifier);
	Location.Position = pos;
	}

Obj::Obj(Equatorial loc) {
	Identifier.push_back("");
	Location = loc;
	}
	
Obj::Obj(string identifier, Equatorial loc) {
	Identifier.push_back(identifier);
	Location = loc;
	}	
	
Obj::Obj(vector<string> identifier, array<double,3> pos) {
	Identifier = identifier;
	Location.Position = pos;
	}
	
Obj::Obj(vector<string> identifier, Equatorial loc) {
	Identifier = identifier;
	Location = loc;
	}	

string Obj::getIdentifier(int number) {
	if (Identifier.size() > number) {
		return Identifier[number];
		}
	throw runtime_error("Invalid Identifier number!");	
	}

vector<string> Obj::getIdentifier() {
	return Identifier;
	}
	
void Obj::addIdentifier(string identifier) {
	Identifier.push_back(identifier);
	}	
	
void Obj::addIdentifier(vector<string> identifier) {
	for (vector<string>::iterator iter = identifier.begin(); iter < identifier.end(); iter++) {
		Identifier.push_back(*iter);
		}
	}	
	
Equatorial Obj::getLocation() {
	return Location;
	}			
	
double Obj::radialComovingDistance(const Universe& u) { 
	return Location.radialComovingDistance(u);
	}

double Obj::transverseComovingDistance(const Universe& u) { 
	return Location.transverseComovingDistance(u);		
	}
	
double Obj::angularDiameterDistance(const Universe& u) { 
	return Location.angularDiameterDistance(u);			
	}

double Obj::luminosityDistance(const Universe& u) { 
	return Location.luminosityDistance(u);			
	}		

double Obj::lookBackTime(const Universe& u) { 
	return Location.lookBackTime(u);
	}
