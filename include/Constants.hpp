#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <complex>

using namespace std;

extern complex<double> complexZero;
extern complex<double> complexOne;
extern double pi;
extern double twoPi;
extern double piSq;
extern double e;
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

extern double secPerSiderealDay;

extern double scCadence;
extern double lcCadence;

extern double log2OfE;

extern double log2Pi;
#endif