#ifndef UNIVERSE_HPP
#define UNIVERSE_HPP

using namespace std;

/** @file Universe.hpp */


/**
 @brief Class to create universe objects.
 
 @details This class lets you create Friedmann universes with arbitrary values of the standard basic cosmological parameters. The standard cosmological parameters that are user adjustable in this version are - \f$H_{0}\f$ (default: \f$H_{0} = 67.77 \ km/s/Mpc\f$), \f$\Omega_{M}\f$ (default: \f$\Omega_{M} = 0.3071\f$), \f$\Omega_{\Lambda}\f$ (default: \f$\Omega_{\Lambda} = 0.6914\f$), \f$\Omega_{\gamma}\f$ (default: \f$\Omega_{\gamma} = 8.24 \times 10^{-5}\f$), and \f$\Omega_{K}\f$ (default: \f$\Omega_{K} = 0.0014\f$). Universes can be instantated with several different combinations of parameters.
 
 @author Vishal Kasliwal
 @version 0.9.1.0
 @date 02-04-2014
 @pre NIL
 @bug NIL
 @warning Read the documentation
 @copyright Vishal Kasliwal
  */
class Universe {
friend class Spherical;
friend class Equatorial;
friend class Galactic;
friend class Obj;
friend class KeplerObj;
friend bool operator==(const Universe& univ1, const Universe& univ2);
friend bool operator!=(const Universe& univ1, const Universe& univ2);
friend bool operator<(const Universe& univ1, const Universe& univ2);
friend bool operator>(const Universe& univ1, const Universe& univ2);
friend bool operator<=(const Universe& univ1, const Universe& univ2);
friend bool operator>=(const Universe& univ1, const Universe& univ2);
friend ostream &operator<<(ostream& os, const Universe& u);
friend istream &operator>>(istream& is, Universe& u);
private:
	static constexpr double Mpc2Km = 3.08567758e+19, Yrs2Sec = 3.15569e7;
	double H0 = 67.77, OmegaM = (0.022161+0.11889)/(67.77*67.77*0.01*0.01), OmegaL = 0.6914, OmegaG = 8.24e-5, OmegaK = 1.0-(0.022161+0.11889)/(67.77*67.77*0.01*0.01)-0.6914-8.24e-5;	
public:
	static constexpr double c = 299792458.0; /**< \f$c = 299792458.0 \ m/s\f$ - the speed of light. */
	double distH = c/(67.77*1.0e3); /**< \f$d_{H} = c/H_{0}\f$ - the Hubble distance. */
	double timeH = (Mpc2Km/67.77)/(Yrs2Sec*1.0e9); /**< \f$t_{H} = 1/H_{0}\f$ - the Hubble time. */
	double ageH = 0.0; /**< \f$ a_{H}\f$ - the age of the Universe. */
	double sizeH = 0.0; /**< \f$ s_{H}\f$ - the size of the Universe. */
	Universe(/**< */); /**< Creates a default initialized Universe with \f$H_{0} = 67.77 \ km/s/Mpc\f$, \f$ \Omega_{M} = 0.3071 \f$, \f$ \Omega_{\Lambda} = 0.6914 \f$, \f$ \Omega_{\gamma} = 8.24 \times 10^{-5} \f$, \f$ \Omega_{k} = 0.0014 \f$. */
	Universe(double h0 /**< Units: \f$km/s/Mpc\f$ */); /**< Creates a universe with \f$H_{0} = \f$h0\f$ \ km/s/Mpc\f$. All other values are set to the default.*/
	Universe(double omegaM /**< Units: NA*/, double omegaL /**< Units: NA*/); /**< Creates a universe with \f$\Omega_{M} =\f$ omegaM and \f$\Omega_{\Lambda} =\f$ omegaL. All other values are set to the default.*/
	Universe(double h0 /**< Units: \f$km/s/Mpc\f$ */, double omegaM /**< Units: NA*/, double omegaL /**< Units: NA*/);/**< Creates a universe with \f$H_{0} = \f$ h0 \f$km/s/Mpc\f$, \f$\Omega_{M} =\f$ omegaM, and \f$\Omega_{\Lambda} =\f$ omegaL. All other values are set to the default.*/
	Universe(double h0 /**< Units: \f$km/s/Mpc\f$ */, double omegaM /**< Units: NA*/, double omegaL /**< Units: NA*/, double omegaG /**< Units: NA*/); /**< Creates a universe with \f$H_{0} =\f$ h0 \f$km/s/Mpc\f$, \f$\Omega_{M} =\f$ omegaM, \f$\Omega_{\Lambda} =\f$ omegaL, and \f$\Omega_{\gamma} =\f$ omegaG. All other values are set to the default.*/
	Universe(const Universe& orig /**< */); /**< Creates a copy of Universe orig.*/
	Universe& operator=(const Universe& orig /**< Units: NA*/);  /**< Makes universe U equal to universe orig. Two universes are equal if all the cosmological parameters are the same.*/
	~Universe( /**< Units: NA*/) = default; /**< Destroys universe.*/
	bool operator==(const Universe& univ1 /**< Units: NA*/); /**< Checks to see if universe U is equal to universe univ1. Two universes are equal if all the cosmological parameters are the same.*/
	void operator()(const vector<double> &x, vector<double> &dxdt, const double t);
	};
void Hz(const vector<double> &x, vector<double> &dxdt, const double t); /**< This is the standard integrand. Integrating this between 0 and the current value of a gives the distance etc... */
bool operator==(const Universe& univ1, const Universe& univ2);
bool operator!=(const Universe& univ1, const Universe& univ2);
bool operator<(const Universe& univ1, const Universe& univ2);
bool operator>(const Universe& univ1, const Universe& univ2);
bool operator<=(const Universe& univ1, const Universe& univ2);
bool operator>=(const Universe& univ1, const Universe& univ2);
ostream &operator<<(ostream& os, const Universe& u);
istream &operator>>(istream& is, Universe& u);

#endif
