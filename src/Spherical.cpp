#include <cmath>
#include <array>
#include <vector>
#include "boost/numeric/odeint.hpp"
#include "Constants.hpp"
#include "Universe.hpp"
#include "Spherical.hpp"

using namespace std;
using namespace boost::numeric::odeint;

Spherical::Spherical(array<double,3> pos) {
	Position = pos;
	}

double Spherical::getRedShift() {
	return Position[2];
	}

double Spherical::radialComovingDistance(const Universe& u) { 
	vector<double> x(2);
	x[0] = 0.0;
	x[1] = 0.0;
	size_t steps = integrate(u, x, 0.0, Position[2], 0.01);
	return u.distH*x[2];
	}

double Spherical::transverseComovingDistance(const Universe& u) { 
	vector<double> x(2);
	x[0] = 0.0;
	x[1] = 0.0;
	size_t steps = integrate(u, x, 0.0, Position[2], 0.01);
	double thisDistM;
	if (u.OmegaK == 0.0) {
		thisDistM = u.distH*x[0]; 
		}
	else {	
		if (u.OmegaK > 0.0) {
			thisDistM = (u.distH/pow(u.OmegaK,0.5))*sinh(pow(u.OmegaK,0.5)*x[0]);
			}
		else {
			thisDistM = (u.distH/pow(abs(u.OmegaK),0.5))*sin(pow(abs(u.OmegaK),0.5)*x[0]);
			}
		}
	return thisDistM;			
	}
	
double Spherical::angularDiameterDistance(const Universe& u) { 
	vector<double> x(2);
	x[0] = 0.0;
	x[1] = 0.0;
	size_t steps = integrate(u, x, 0.0, Position[2], 0.01);
	double thisDistM;
	if (u.OmegaK == 0.0) {
		thisDistM = u.distH*x[0]; 
		}
	else {	
		if (u.OmegaK > 0.0) {
			thisDistM = (u.distH/pow(u.OmegaK,0.5))*sinh(pow(u.OmegaK,0.5)*x[0]);
			}
		else {
			thisDistM = (u.distH/pow(abs(u.OmegaK),0.5))*sin(pow(abs(u.OmegaK),0.5)*x[0]);
			}
		}
	return thisDistM/(1+Position[2]);			
	}

double Spherical::luminosityDistance(const Universe& u) { 
	vector<double> x(2);
	x[0] = 0.0;
	x[1] = 0.0;
	size_t steps = integrate(u, x, 0.0, Position[0], 0.01);
	double thisDistM;
	if (u.OmegaK == 0.0) {
		thisDistM = u.distH*x[0]; 
		}
	else {	
		if (u.OmegaK > 0.0) {
			thisDistM = (u.distH/pow(u.OmegaK,0.5))*sinh(pow(u.OmegaK,0.5)*x[0]);
			}
		else {
			thisDistM = (u.distH/pow(abs(u.OmegaK),0.5))*sin(pow(abs(u.OmegaK),0.5)*x[0]);
			}
		}
	return thisDistM*(1+Position[2]);			
	}		

double Spherical::lookBackTime(const Universe& u) { 
	vector<double> x(2);
	x[0] = 0.0;
	x[1] = 0.0;
	size_t steps = integrate(u, x, 0.0, Position[2], 0.01);
	return u.timeH*x[1];
	}
	
bool operator==(const Spherical& sph1, const Spherical& sph2) {
	bool val = false;
	if (sph1.Position == sph2.Position) {
		val = true;
		}
	return val;
	}
	
bool operator!=(const Spherical& sph1, const Spherical& sph2) {
	bool val = true;
	if (!(sph1==sph2)) {
		val = false;
		}
	return val;
	}	

Equatorial::Equatorial(array<double,3> pos) {
	Position = pos;
	}
	
Galactic::Galactic(array<double,3> pos) {
	Position = pos;
	}				
