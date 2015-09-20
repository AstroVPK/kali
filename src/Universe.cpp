#include <cmath>
#include "boost/numeric/odeint.hpp"
#include "Universe.hpp"

using namespace std;
using namespace boost::numeric::odeint;

Universe::Universe() {
	H0 = 0.6777;
	OmegaM = 0.141051/(67.77*67.77*0.01*0.01);
	OmegaL = 0.6914;
	OmegaG = 8.24e-5;
	OmegaK = 1.0-OmegaM-OmegaL-OmegaG;
	distH = c/(67.77*1.0e3);
	timeH = (Mpc2Km/67.77)/(Yrs2Sec*1.0e9);
	vector<double> initX(6);
	initX[0] = 0.0; initX[1] = 0.0; initX[2] = OmegaM; initX[3] = OmegaL; initX[4] = OmegaG; initX[5] = OmegaK;
	size_t steps = integrate(Hz, initX, 0.0, 1.0e+38, 1.0e-6);
	sizeH = initX[0]*distH;
	ageH = initX[1]*timeH;
	}

Universe::Universe(double h0) {
	H0 = h0*0.01;
	OmegaM = 0.141051/(h0*h0*0.01*0.01);
	OmegaL = 0.6914;
	OmegaG = 8.24e-5;
	OmegaK = 1.0-OmegaM-OmegaL-OmegaG;
	distH = c/(h0*1.0e3);
	timeH = (Mpc2Km/h0)/(Yrs2Sec*1.0e9);
	vector<double> initX(6);
	initX[0] = 0.0; initX[1] = 0.0; initX[2] = OmegaM; initX[3] = OmegaL; initX[4] = OmegaG; initX[5] = OmegaK;
	size_t steps = integrate(Hz, initX, 0.0, 1.0e+38, 1.0e-6);
	sizeH = initX[0]*distH;
	ageH = initX[1]*timeH;
	}
	
Universe::Universe(double omegaM, double omegaL) {
	H0 = 67.77;
	OmegaM = omegaM;
	OmegaL = omegaL;
	OmegaG = 8.24e-5;
	OmegaK = 1.0-OmegaM-OmegaL-OmegaG;
	distH = c/(H0*1e3);
	timeH = (Mpc2Km/H0)/(Yrs2Sec*1.0e9);
	vector<double> initX(6);
	initX[0] = 0.0; initX[1] = 0.0; initX[2] = OmegaM; initX[3] = OmegaL; initX[4] = OmegaG; initX[5] = OmegaK;
	size_t steps = integrate(Hz, initX, 0.0, 1.0e+38, 1.0e-6);
	sizeH = initX[0]*distH;
	ageH = initX[1]*timeH;
	}
	
Universe::Universe(double h0, double omegaM, double omegaL) {
	H0 = h0;
	OmegaM = omegaM;
	OmegaL = omegaL;
	OmegaG = 8.24e-5;
	OmegaK = 1.0-OmegaM-OmegaL-OmegaG;
	distH = c/(h0*1.0e3);
	timeH = (Mpc2Km/h0)/(Yrs2Sec*1.0e9);
	vector<double> initX(6);
	initX[0] = 0.0; initX[1] = 0.0; initX[2] = OmegaM; initX[3] = OmegaL; initX[4] = OmegaG; initX[5] = OmegaK;
	size_t steps = integrate(Hz, initX, 0.0, 1.0e+38, 1.0e-6);
	sizeH = initX[0]*distH;
	ageH = initX[1]*timeH;
	}

Universe::Universe(double h0, double omegaM, double omegaL, double omegaG) {
	H0 = h0;
	OmegaM = omegaM;
	OmegaL = omegaL;
	OmegaG = omegaG;
	OmegaK = 1.0-OmegaM-OmegaL-OmegaG;
	distH = c/(h0*1.0e3);
	timeH = (Mpc2Km/h0)/(Yrs2Sec*1.0e9);
	vector<double> initX(6);
	initX[0] = 0.0; initX[1] = 0.0; initX[2] = OmegaM; initX[3] = OmegaL; initX[4] = OmegaG; initX[5] = OmegaK;
	size_t steps = integrate(Hz, initX, 0.0, 1.0e+38, 1.0e-6);
	sizeH = initX[0]*distH;
	ageH = initX[1]*timeH;
	}

Universe::Universe(const Universe& orig) {
	H0 = orig.H0;
	OmegaM = orig.OmegaM;
	OmegaL = orig.OmegaL;
	OmegaG = orig.OmegaG;
	OmegaK = orig.OmegaK;
	distH = orig.distH;
	timeH = orig.timeH;
	
	ageH = orig.ageH;
	}
	
Universe& Universe::operator=(const Universe& orig) {
	H0 = orig.H0;
	OmegaM = orig.OmegaM;
	OmegaL = orig.OmegaL;
	OmegaG = orig.OmegaG;
	OmegaK = orig.OmegaK;
	distH = orig.distH;
	timeH = orig.timeH;
	sizeH = orig.sizeH;
	ageH = orig.ageH;
	return *this;
	}					

void Universe::operator()(const vector<double> &x, vector<double> &dxdt, const double t) {
	dxdt[0] = 1.0/pow(OmegaL + pow(1.0+t,3.0)*OmegaM + pow(1.0+t,4.0)*OmegaG + pow(1.0+t,2.0)*OmegaK, 0.5);
	dxdt[1] = 1.0/((1.0+t)*pow(OmegaL + pow(1.0+t,3.0)*OmegaM + pow(1.0+t,4.0)*OmegaG + pow(1.0+t,2.0)*OmegaK, 0.5));
	}
	
void Hz(const vector<double> &x, vector<double> &dxdt, const double t) {
	double omegaM = x[2], omegaL = x[3], omegaG = x[4], omegaK = x[5];
	dxdt[0] = 1.0/pow(omegaL + pow(1.0+t,3.0)*omegaM + pow(1.0+t,4.0)*omegaG + pow(1.0+t,2.0)*omegaK, 0.5);
	dxdt[1] = 1.0/((1.0+t)*pow(omegaL + pow(1.0+t,3.0)*omegaM + pow(1.0+t,4.0)*omegaG + pow(1.0+t,2.0)*omegaK, 0.5));
	dxdt[2] = 0.0;
	dxdt[3] = 0.0;
	dxdt[4] = 0.0;
	dxdt[5] = 0.0;
	}
	
bool operator==(const Universe& univ1, const Universe& univ2) {
	bool val = false;
	if ((univ1.H0 == univ2.H0) and (univ1.OmegaM == univ2.OmegaM) and (univ1.OmegaL == univ2.OmegaL) and (univ1.OmegaG == univ2.OmegaG)) {
		val = true;
		}
	return val;
	}
	
bool operator!=(const Universe& univ1, const Universe& univ2) {
	bool val = true;
	if (!(univ1==univ2)) {
		val = false;
		}
	return val;
	}
	
bool operator<(const Universe& univ1, const Universe& univ2) {
	bool val = false;
	if (univ1.sizeH < univ2.sizeH) {
		val = true;
		}
	return val;
	}
bool operator>(const Universe& univ1, const Universe& univ2) {
	bool val = true;
	if (!(univ1 <= univ2)) {
		val = false;
		}
	return val;
	}
bool operator<=(const Universe& univ1, const Universe& univ2) {
	bool val = false;
	if (univ1.sizeH <= univ2.sizeH) {
		val = true;
		}
	return val;
	}
bool operator>=(const Universe& univ1, const Universe& univ2) {
	bool val = true;
	if (!(univ1 < univ2)) {
		val = false;
		}
	return val;
	}
	
ostream &operator<<(ostream& os, const Universe& u) {
	os << "H0: " << u.H0 << " (km/s/Mpc); OmegaM: " << u.OmegaM << "; OmegaL: " << u.OmegaL << "; OmegaG: " << u.OmegaG << "; OmegaK: " << u.OmegaK << "; Size: " << u.sizeH << "(Mpc); Age: " << u.ageH << "(Gyr);";
	return os;  
	}
	
istream &operator>>(istream& is, Universe& u) {
	double h0, omegaM, omegaL, omegaG;
	cout << "H0 (km/s/Mpc): ";
	is >> h0;
	cout << "OmegaM: ";
	cin >> omegaM;
	cout << "OmegaL: ";
	cin >> omegaL;
	cout << "OmegaG: ";
	cin >> omegaG;
		if (is) {
			u = Universe(h0, omegaM, omegaL, omegaG);
			}
		else {
			u = Universe();
			}	
	return is;
	}		
