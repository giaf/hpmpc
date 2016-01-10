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

#include "/opt/hpmpc/include/c_interface.h"



// the gateway function 
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
	{

		
	// get data 
	double *A, *G, *C, *f, *R, *Q, *Qf, *r, *q, *qf, *y, *x0, *L0, *xe, *Le, *w, *lam;
	
	const int alg = (int) mxGetScalar(prhs[0]);
	const int nx = (int) mxGetScalar(prhs[1]);
	const int nw = (int) mxGetScalar(prhs[2]);
	const int ny = (int) mxGetScalar(prhs[3]);
	const int N  = (int) mxGetScalar(prhs[4]);

	A = mxGetPr(prhs[5]);
	G = mxGetPr(prhs[6]);
	C = mxGetPr(prhs[7]);
	f = mxGetPr(prhs[8]);
	R = mxGetPr(prhs[9]);
	Q = mxGetPr(prhs[10]);
	r = mxGetPr(prhs[11]);
	q = mxGetPr(prhs[12]);
	y = mxGetPr(prhs[13]);
	x0 = mxGetPr(prhs[14]);
	L0 = mxGetPr(prhs[15]);
	xe = mxGetPr(prhs[16]);
	Le = mxGetPr(prhs[17]);
	w = mxGetPr(prhs[18]);
	lam = mxGetPr(prhs[19]);

	if(alg==2)
		{
		Qf = Q+N*nx*nx;
		qf = q+N*nx;
		}
	else //if(alg==0 || alg==1)
		{
		Qf = Q+N*ny*ny;
		qf = q+N*ny;
		}
	
	int work_space_size = hpmpc_ric_mhe_if_dp_work_space(nx, nw, ny, N);

	double *work = (double *) malloc( work_space_size * sizeof(double) );

//	int ii;
//	for(ii=0; ii<work_space_size; ii++)
//		work[ii] = 0;
	
	// call the solver
	int hpmpc_status = fortran_order_riccati_mhe_if( 'd', alg, nx, nw, ny, N, A, G, C, f, R, Q, Qf, r, q, qf, y, x0, L0, xe, Le, w, lam, work);
	//int hpmpc_status = c_order_riccati_mhe_if( 'd', alg, nx, nw, ny, N, A, G, C, f, R, Q, Qf, r, q, qf, y, x0, L0, xe, Le, w, lam, work);

	free(work);

	//printf("\nout of wrapper\n");
		
	return;

	}

