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

#include <hpmpc/c_interface.h>



// the gateway function 
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
	{
		
	// get data 
	double *A, *B, *b, *Q, *Qf, *R, *S, *q, *qf, *r, *x, *u, *pi;
	
	const int nx = (int) mxGetScalar(prhs[0]);
	const int nu = (int) mxGetScalar(prhs[1]);
	const int N  = (int) mxGetScalar(prhs[2]);

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
	
	int work_space_size = hpmpc_ric_mpc_dp_work_space(nx, nu, N);

	double *work = (double *) malloc( work_space_size * sizeof(double) );

	// call the solver
	fortran_order_riccati_mpc( 'd', nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, x, u, pi, work );
	//c_order_riccati_mpc( 'd', nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, x, u, pi, work );
	
	free(work);

	return;

	}

