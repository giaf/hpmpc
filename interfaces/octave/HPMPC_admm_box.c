/**************************************************************************************************
* 
* Author: Gianluca Frison, giaf@imm.dtu.dk
*
* Factorizes in double precision the extended LQ control problem, factorized algorithm
*
**************************************************************************************************/

#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
/*#include <math.h>*/

/*#include "hpmpc/aux_d.h"*/
/*#include "/include/hpmpc/block_size.h"*/
/*#include "hpmpc/mpc_solvers.h"*/

//void fortran_order_dynamic_mem_admm_soft_wrapper( int k_max, double tol, double rho, double alpha, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double *T, double* lb, double* ub, double* x, double* u, int* nIt, double *stat );
int fortran_order_admm_box_mpc( int k_max, double tol, double rho, double alpha, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lb, double* ub, double* x, double* u, int* nIt, double *stat );




// the gateway function 
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
	{
		
	// get data 
	int k_max;
	double tol, rho, alpha, *A, *B, *b, *Q, *Qf, *R, *S, *q, *qf, *r, *x, *u, *lb, *ub, *stat, *kkk;
	
	k_max = (int) mxGetScalar(prhs[0]);
	tol = mxGetScalar(prhs[1]);
	rho = mxGetScalar(prhs[2]);
	alpha = mxGetScalar(prhs[3]);
	const int nx = (int) mxGetScalar(prhs[4]);
	const int nu = (int) mxGetScalar(prhs[5]);
	const int N  = (int) mxGetScalar(prhs[6]);
/*	nb  = (int) mxGetScalar(prhs[5]);*/

	A = mxGetPr(prhs[7]);
	B = mxGetPr(prhs[8]);
	b = mxGetPr(prhs[9]);
	Q = mxGetPr(prhs[10]);
	Qf = mxGetPr(prhs[11]);
	R = mxGetPr(prhs[12]);
	S = mxGetPr(prhs[13]);
	q = mxGetPr(prhs[14]);
	qf = mxGetPr(prhs[15]);
	r = mxGetPr(prhs[16]);
	lb = mxGetPr(prhs[17]);
	ub = mxGetPr(prhs[18]);
	x = mxGetPr(prhs[19]);
	u = mxGetPr(prhs[20]);
	kkk = mxGetPr(prhs[21]);
	stat = mxGetPr(prhs[22]);
	
	int kk = -1;
	
	// parameters
//	double rho = 10.0; // penalty parameter
//	double alpha = 1.9; // relaxation parameter

	// call solver 

/*printf("\nstart of solver\n");*/

//	fortran_order_admm_soft_wrapper(k_max, tol, rho, alpha, nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, Z, z, lb, ub, x, u, &kk, stat );
	fortran_order_admm_box_mpc(k_max, tol, rho, alpha, nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, x, u, &kk, stat);
//	fortran_order_dynamic_mem_admm_soft_wrapper(k_max, tol, rho, alpha, nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, T, lb, ub, x, u, &kk, stat);
/*	fortran_order_static_mem_ip_wrapper(k_max, tol, nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, x, u, &kk, stat);*/
	
/*printf("\nend of solver\n");*/

	*kkk = (double) kk;

	return;

	}

