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

int fortran_order_dynamic_mem_riccati_wrapper_mhe( const int nx, const int nw, const int ny, const int N, double *A, double *G, double *C, double *f, double *Q, double *R, double *q, double *r, double *y, double *x0, double *L0, double *xe, double *Le );



/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
	{
		
	/* get data */
	double *A, *G, *C, *f, *Q, *R, *q, *r, *y, *x0, *L0, *xe, *Le;
	
	const int nx = (int) mxGetScalar(prhs[0]);
	const int nw = (int) mxGetScalar(prhs[1]);
	const int ny = (int) mxGetScalar(prhs[2]);
	const int N  = (int) mxGetScalar(prhs[3]);

	A = mxGetPr(prhs[4]);
	G = mxGetPr(prhs[5]);
	C = mxGetPr(prhs[6]);
	f = mxGetPr(prhs[7]);
	Q = mxGetPr(prhs[8]);
	R = mxGetPr(prhs[9]);
	q = mxGetPr(prhs[10]);
	r = mxGetPr(prhs[11]);
	y = mxGetPr(prhs[12]);
	x0 = mxGetPr(prhs[13]);
	L0 = mxGetPr(prhs[14]);
	xe = mxGetPr(prhs[15]);
	Le = mxGetPr(prhs[16]);
	
	double *work;
	
	fortran_order_dynamic_mem_riccati_wrapper_mhe( nx, nw, ny, N, A, G, C, f, Q, R, q, r, y, x0, L0, xe, Le );

	return;

	}

