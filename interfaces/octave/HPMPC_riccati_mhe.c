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
	double *A, *G, *C, *f, *Q, *R, *q, *r, *y, *x0, *L0, *xe, *Le, *w;
	
	const int smooth = (int) mxGetScalar(prhs[0]);
	const int nx = (int) mxGetScalar(prhs[1]);
	const int nw = (int) mxGetScalar(prhs[2]);
	const int ny = (int) mxGetScalar(prhs[3]);
	const int N  = (int) mxGetScalar(prhs[4]);

	A = mxGetPr(prhs[5]);
	G = mxGetPr(prhs[6]);
	C = mxGetPr(prhs[7]);
	f = mxGetPr(prhs[8]);
	Q = mxGetPr(prhs[9]);
	R = mxGetPr(prhs[10]);
	q = mxGetPr(prhs[11]);
	r = mxGetPr(prhs[12]);
	y = mxGetPr(prhs[13]);
	x0 = mxGetPr(prhs[14]);
	L0 = mxGetPr(prhs[15]);
	xe = mxGetPr(prhs[16]);
	Le = mxGetPr(prhs[17]);
	w = mxGetPr(prhs[18]);
	
	int work_space_size = hpmpc_ric_mhe_dp_work_space(nx, nw, ny, N);

	double *work = (double *) malloc( work_space_size * sizeof(double) );
	
	// call the solver
	fortran_order_riccati_mhe( 'd', smooth, nx, nw, ny, N, A, G, C, f, Q, R, q, r, y, x0, L0, xe, Le, w, work );
	//c_order_riccati_mhe( 'd', smooth, nx, nw, ny, N, A, G, C, f, Q, R, q, r, y, x0, L0, xe, Le, w, work );

	free(work);

	return;

	}

