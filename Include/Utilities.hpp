#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <mkl_types.h>

using namespace std;

tuple<vector<double>,vector<int>> histogram(vector<double> data, int numBins);
tuple<vector<double>,vector<int>> histogram(vector<double> data, int numBins, double base);
tuple<vector<double>,vector<int>> histogram(vector<double> data, int numBins, string base);
bool exists(const string& filename);
int gcd(int a, int b);
int lcm(int a, int b);
#endif
