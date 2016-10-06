#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <complex>

using namespace std;

namespace kali {

extern complex<double> complexZero;
extern complex<double> complexOne;
extern double pi;
extern double twoPi;
extern double piSq;
extern double fourPiSq;
extern double halfPi;
extern double log2Pi;
extern double e;
extern double log2OfE;
extern double infiniteVal;

extern double G;
extern double c;
extern double AU;
extern double Parsec;
extern double Day;
extern double Year;
extern double kms2ms;
extern double SolarMass;
extern double SolarMassPerCubicParsec;
extern double secPerSiderealDay;

namespace Kepler {

extern double integrationTime;
extern double readTime;
extern int numIntegrationsSC;
extern int numIntegrationsLC;

extern double samplingIntervalSC;
extern double samplingIntervalLC;

extern double samplingFrequencySC;
extern double samplingFrequencyLC;

extern double NyquistFrequencySC;
extern double NyquistFrequencyLC;


extern double scCadence;
extern double lcCadence;
} // namespace Kepler

} // namespace kali

#endif
