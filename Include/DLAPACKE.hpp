#ifndef DLAPACKE_HPP
#define DLAPACKE_HPP

using namespace std;

double DLAPACKE_dgzthp(int N, double* R, double* X, double* Y); // Double General toeplitZ TrencH Precompute

double DLAPACKE_dgzthi(int i, int j, int N, double* X); // Double General toeplitZ TrencH Inverse

double DLAPACKE_dszthp(int N, double* R, double* X, double* Y); // Double Symmetric toeplitZ TrencH Precompute

double DLAPACKE_dszthi(int i, int j, int N, double* X); // Double Symmetric toeplitZ TrencH Inverse

/*double zgzthp(int N, double* R, double* X, double* Y); // double complex (Z) General toeplitZ TrencH Precompute

double zgzthi(int N, double* R, double* X, double* Y); // double complex (Z) General toeplitZ TrencH Inverse

double zhzthp(int N, double* R, double* X, double* Y); // double complex (Z) Hermitian toeplitZ TrencH Precompute

double zhzthi(int N, double* R, double* X, double* Y); // double complex (Z) Hermitian toeplitZ TrencH Inverse*/

/*double DLAPACKE_dgzthi_recursive(int i, int j, int N, double* X); // Double General toeplitZ TrencH Recursive Inverse

double DLAPACKE_dszthi_recursive(int i, int j, int N, double* X); // Double Symmetric toeplitZ TrencH Recursive Inverse*/

#endif
