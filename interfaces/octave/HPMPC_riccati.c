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

int fortran_order_dynamic_mem_riccati_wrapper_init( const int nx, const int nu, const int N, double *A, double *B, double *b, double *Q, double *Qf, double *S, double *R, double *q, double *qf, double *r, double **ptr_work );
int fortran_order_dynamic_mem_riccati_wrapper_free( double *work );
int fortran_order_dynamic_mem_riccati_wrapper_fact_solve( const int nx, const int nu, const int N, double *x, double *u, double *pi, double *work );




/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
	{
		
	/* get data */
	double *A, *B, *b, *Q, *Qf, *R, *S, *q, *qf, *r, *x, *u, *pi;
	
	const int nx = (int) mxGetScalar(prhs[0]);
	const int nu = (int) mxGetScalar(prhs[1]);
	const int N  = (int) mxGetScalar(prhs[2]);
/*	nb  = (int) mxGetScalar(prhs[5]);*/

	A = mxGetPr(prhs[3]);
	B = mxGetPr(prhs[4]);
	b = mxGetPr(prhs[5]);
	Q = mxGetPr(prhs[6]);
	Qf = mxGetPr(prhs[7]);
	R = mxGetPr(prhs[8]);
	S = mxGetPr(prhs[9]);
	q = mxGetPr(prhs[10]);
	qf = mxGetPr(prhs[11]);
	r = mxGetPr(prhs[12]);
	x = mxGetPr(prhs[13]);
	u = mxGetPr(prhs[14]);
	pi = mxGetPr(prhs[15]);
	
	double *work;
	
	fortran_order_dynamic_mem_riccati_wrapper_init( nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, &work );
	
	fortran_order_dynamic_mem_riccati_wrapper_fact_solve( nx, nu, N, x, u, pi, work );
/*	printf("\nfact and solve\n\n");*/
	
/*	fortran_order_dynamic_mem_riccati_wrapper_solve( nx, nu, N, q, qf, r, x, u, pi, work );*/
/*	printf("\ntri solve\n\n");*/

	fortran_order_dynamic_mem_riccati_wrapper_free( work );

	return;

	}

