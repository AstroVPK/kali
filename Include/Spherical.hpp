#ifndef SPHERICAL_HPP
#define SPHERICAL_HPP

#include "Constants.hpp"
#include "Universe.hpp"

using namespace std;

inline double d2r(double degrees) {
	return degrees*(pi/(double)180.0);	
	}
	
inline double r2d(double radians) {
	return radians*((double)180.0/pi);	
	}	

class Spherical {
friend class Obj;
friend class KeplerObj;
friend bool operator==(const Spherical& sph1, const Spherical& sph2);
friend bool operator!=(const Spherical& sph1, const Spherical& sph2);
protected:
	array<double,3> Position;
public:
	Spherical() = default;
	Spherical(array<double,3> pos);
	double getRedShift();
	double radialComovingDistance(const Universe& u);
	double transverseComovingDistance(const Universe& u);
	double angularDiameterDistance(const Universe& u);	
	double luminosityDistance(const Universe& u);
	double lookBackTime(const Universe& u);
	};	
bool operator==(const Spherical& sph1, const Spherical& sph2);
bool operator!=(const Spherical& sph1, const Spherical& sph2);	
		
class Equatorial: public Spherical {
public:
	Equatorial() = default;
	Equatorial(array<double,3> pos);
	};
	
class Galactic: public Spherical {
public:
	Galactic() = default;
	Galactic(array<double,3> pos);
	};	
	
#endif	
