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

void fortran_order_dynamic_mem_ip_wrapper( int k_max, double tol, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lb, double* ub, double* x, double* u, int* nIt, double *stat );



/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
	{
		
	/* get data */
	int k_max;
	double tol, *A, *B, *b, *Q, *Qf, *R, *S, *q, *qf, *r, *x, *u, *lb, *ub, *stat, *kkk;
	
	k_max = (int) mxGetScalar(prhs[0]);
	tol = mxGetScalar(prhs[1]);
	const int nx = (int) mxGetScalar(prhs[2]);
	const int nu = (int) mxGetScalar(prhs[3]);
	const int N  = (int) mxGetScalar(prhs[4]);
/*	nb  = (int) mxGetScalar(prhs[5]);*/

	A = mxGetPr(prhs[5]);
	B = mxGetPr(prhs[6]);
	b = mxGetPr(prhs[7]);
	Q = mxGetPr(prhs[8]);
	Qf = mxGetPr(prhs[9]);
	R = mxGetPr(prhs[10]);
	S = mxGetPr(prhs[11]);
	q = mxGetPr(prhs[12]);
	qf = mxGetPr(prhs[13]);
	r = mxGetPr(prhs[14]);
	lb = mxGetPr(prhs[15]);
	ub = mxGetPr(prhs[16]);
	x = mxGetPr(prhs[17]);
	u = mxGetPr(prhs[18]);
	kkk = mxGetPr(prhs[19]);
	stat = mxGetPr(prhs[20]);
	
	int kk = -1;
	
/*	 call solver */
	fortran_order_dynamic_mem_ip_wrapper(k_max, tol, nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, x, u, &kk, stat);
/*	fortran_order_static_mem_ip_wrapper(k_max, tol, nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, x, u, &kk, stat);*/
	
	*kkk = (double) kk;

	return;

	}

