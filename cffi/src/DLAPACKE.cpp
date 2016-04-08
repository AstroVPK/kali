#define DIAGNOSTIC 1
#if DIAGNOSTIC>0
#include <iostream>
#include <ctime>
#endif
#include <cmath>
#include "DLAPACKE.hpp"

using namespace std;
	
double DLAPACKE_dgzthp(int N, double* R, double* X, double* Y) {
	double e, g, lnd;
	X[N] = (double)1.0-R[N-1]*R[N+1];
	X[N+1] = -R[N-1];
	X[N-1] = -R[N+1];
	lnd = log(X[N]);
	for (int i = 1; i < N; i++) {
		e = R[N-i-1];
		g = R[N+i+1];
		for (int j = 1; j < i+1; j++) {
			e += X[N+j]*R[N-i+j-1];
			g += R[N+j]*X[N-i+j-1];
			}
		e /= -X[N];
		g /= -X[N];
		X[N+i+1] = e;
		X[N-i-1] = g;
		Y[N] = -e*g*X[N];
		for (int j = 1; j < i+1; j++) {
			Y[N+j]=X[N-i+j-1]*e;
			Y[N-j]=X[N+i-j+1]*g;
			}
		for (int j = 1; j < i+1; j++) {
			X[N+j] += Y[N+j];
			X[N-j] += Y[N-j];
			}
		X[N] += Y[N];
		lnd += log(X[N]);
		}
	return lnd;
	}
	
double DLAPACKE_dgzthi(int i, int j, int N, double* X) {
	double val = 0.0, sum;
	int iPos, jPos;
	if (i+j > N) {
		val = DLAPACKE_dgzthi(N-j, N-i, N, X);
		}
	else {
		if ((i == 0) and (j == 0)) {
			val = 1.0/X[N];
			}
		else {
			if ((i == 0) and (j > 0)) {
				val = X[N+j]/X[N];
				}
			else {
				if ((i > 0) and (j == 0)) {
					val = X[N-i]/X[N];
					}
				else {
					sum = 0.0;
					iPos = i, jPos = j;
					for (int decCounter = fmin(i,j); decCounter > 0; decCounter--) {
						sum += ((X[N-iPos]*X[N+jPos] - X[2*N+1-iPos]*X[jPos-1])/X[N]);
						iPos -= 1;
						jPos -= 1;
						}
					if ((iPos > 0) and (jPos == 0)) {
						val = (X[N-iPos]/X[N]) + sum;
						}
					else {
						if ((iPos == 0) and (jPos > 0)) {
							val = (X[N+jPos]/X[N]) + sum; 
							}
						else {
								val = (1.0/X[N]) + sum;
								}
						}
					}
				}
			}
		}
	return val;
	}

double DLAPACKE_dszthp(int N, double* R, double* X, double* Y) {
	double g, lnd;
	X[0] = 1.0-R[1]*R[1];
	X[1] = -R[1];
	lnd = log(X[0]);
	for (int i = 1; i < N; i++) {
		g = R[i+1];
		for (int j = 1; j < i+1; j++) {
			g += R[j]*X[i+1-j];
			}
		g /= -X[0];
		X[i+1] = g;
		Y[0] = -g*g*X[0];
		for (int j = 1; j < i+1; j++) {
			Y[j]=X[i-j+1]*g;
			}
		for (int j = 1; j < i+1; j++) {
			X[j] += Y[j];
			}
		X[0] += Y[0];
		lnd += log(X[0]);
		}
	return lnd;
	}

double DLAPACKE_dszthi(int i, int j, int N, double* X) {
	double val = 0.0, sum;
	int iPos, jPos;
	if (i+j > N) {
		val = DLAPACKE_dszthi(N-j, N-i, N, X);
		}
	else {
		if (j < i) {
			val = DLAPACKE_dszthi(j, i, N, X);
			}
		else {
			if ((i == 0) and (j == 0)) {
				val = 1.0/X[0];
				}
			else {
				if ((i == 0) and (j > 0)) {
					val = X[j]/X[0];
					}
				else {
					sum = 0.0;
					iPos = i, jPos = j;
					for (int decCounter = fmin(i,j); decCounter > 0; decCounter--) {
						sum += ((X[iPos]*X[jPos] - X[N+1-iPos]*X[N+1-jPos])/X[0]);
						iPos -= 1;
						jPos -= 1;
						}
					if ((iPos == 0) and (jPos == 0)) {
						val = (1.0/X[0]) + sum;
						}
					else {
						val = (X[jPos]/X[0]) + sum; 
						}
					}
				}
			}
		}
	return val;
	}

/*double DLAPACKE_dgzthi_recursive(int i, int j, int N, double* X) {
	double val = 0.0;
	if (i+j > N) {
		val = dgzthi_recursive(N-j, N-i, N, X);
		}
	else {	
		if ((i == 0) and (j == 0)) {
			val = 1.0/X[N];
			}
		else {
			if ((i == 0) and (j > 0)) {
				val = X[N+j]/X[N];
				}
			else {	
				if ((i > 0) and (j == 0)) {
					val = X[N-i]/X[N];
					}
				else {
					val = dgzthi_recursive(i-1, j-1, N, X) + ((X[N-i]*X[N+j] - X[2*N+1-i]*X[j-1])/X[N]);
					}
				}
			}
		}					
	return val;
	}	
	
double DLAPACKE_dszthi_recursive(int i, int j, int N, double* X) {
	double val = 0.0;
	if (i+j > N) {
		val = dszthi_recursive(N-j, N-i, N, X);
		}
	else {	
		if (j < i) {
			val = dszthi_recursive(j, i, N, X);	
			}
		else {	
			if ((i == 0) and (j == 0)) {
				val = 1.0/X[0];
				}
			else {
				if ((i == 0) and (j > 0)) {
					val = X[j]/X[0];
					}
				else {
					val = dszthi_recursive(i-1, j-1, N, X) + ((X[i]*X[j] - X[N+1-i]*X[N+1-j])/X[0]);
					}
				}
			}	
		}					
	return val;
	}*/						
	
	 
