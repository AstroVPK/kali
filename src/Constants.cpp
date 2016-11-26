#ifdef __INTEL_COMPILER
    #include <mathimf.h>
    #if defined __APPLE__ && defined __MACH__
        #include <malloc/malloc.h>
    #else
        #include <malloc.h>
    #endif
#else
    #include <math.h>
    #include <mm_malloc.h>
#endif
#include <cmath>
#include <complex>
#include "Constants.hpp"

using namespace std;

extern complex<double> kali::complexZero (0.0,0.0);
extern complex<double> kali::complexOne (1.0,0.0);
extern double kali::pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
extern double kali::twoPi = 2.0*kali::pi;
extern double kali::fourPiSq = 4.0*pow(kali::pi, 2.0);
extern double kali::piSq = pow(kali::pi, 2.0);
extern double kali::halfPi = kali::pi/2.0;
extern double kali::e = 2.71828182845904523536028747135266249775724709369995;
extern double kali::infiniteVal = HUGE_VAL;
extern double kali::log2OfE = log2(kali::e);
extern double kali::log2Pi = log2(2.0*kali::pi)/kali::log2OfE;

extern double kali::G = 6.67408e-11; // m^3/kg s^2
extern double kali::c = 299792458.0; // m/s
extern double kali::AU = 1.4960e11; // m
extern double kali::Parsec = 3.0857e16; // m
extern double kali::Day = 86164.090530833; //s
extern double kali::Year = 31557600.0; // s
extern double kali::kms2ms = 1.0e3; // m/s
extern double kali::SolarMass = 1.98855e30; // kg
extern double kali::SolarMassPerCubicParsec = kali::SolarMass/pow(kali::Parsec, 3.0); // kg/m^3
extern double kali::secPerSiderealDay = 86164.090530833;

extern double kali::Kepler::integrationTime = 6.019802903;
extern double kali::Kepler::readTime = 0.5189485261;
extern int kali::Kepler::numIntegrationsSC = 9;
extern int kali::Kepler::numIntegrationsLC = 270;

extern double kali::Kepler::samplingIntervalSC = (kali::Kepler::integrationTime + kali::Kepler::readTime)*kali::Kepler::numIntegrationsSC;
extern double kali::Kepler::samplingIntervalLC = (kali::Kepler::integrationTime + kali::Kepler::readTime)*kali::Kepler::numIntegrationsLC;

extern double kali::Kepler::samplingFrequencySC = 1.0/((kali::Kepler::integrationTime + kali::Kepler::readTime)*kali::Kepler::numIntegrationsSC);
extern double kali::Kepler::samplingFrequencyLC = 1.0/((kali::Kepler::integrationTime + kali::Kepler::readTime)*kali::Kepler::numIntegrationsLC);

extern double kali::Kepler::NyquistFrequencySC = kali::Kepler::samplingFrequencySC/2.0;
extern double kali::Kepler::NyquistFrequencyLC = kali::Kepler::samplingFrequencyLC/2.0;


//extern double scCadence = ((integrationTime+readTime)*numIntegrationsSC)/((double)3600.0*(double)23.9344696);
//extern double lcCadence = ((integrationTime+readTime)*numIntegrationsLC)/((double)3600.0*(double)23.9344696);

extern double kali::Kepler::scCadence = (kali::Kepler::integrationTime + kali::Kepler::readTime)*kali::Kepler::numIntegrationsSC;
extern double kali::Kepler::lcCadence = (kali::Kepler::integrationTime + kali::Kepler::readTime)*kali::Kepler::numIntegrationsLC;
