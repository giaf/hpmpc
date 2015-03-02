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
	int k_max;
	double mu0, tol, *A, *B, *b, *Q, *Qf, *R, *S, *q, *qf, *r, *x, *u, *lb, *ub, *C, *D, *lg, *ug, *CN, *lgN, *ugN, *stat, *kkk, *inf_norm_res, *pi, *lam, *t;
	
	kkk   = mxGetPr(prhs[0]);
	k_max = (int) mxGetScalar(prhs[1]);
	mu0   = mxGetScalar(prhs[2]);
	tol   = mxGetScalar(prhs[3]);
	const int N   = (int) mxGetScalar(prhs[4]);
	const int nx  = (int) mxGetScalar(prhs[5]);
	const int nu  = (int) mxGetScalar(prhs[6]);
	const int nb  = (int) mxGetScalar(prhs[7]);
	const int ng  = (int) mxGetScalar(prhs[8]);
	const int ngN = (int) mxGetScalar(prhs[9]);

	A    = mxGetPr(prhs[10]);
	B    = mxGetPr(prhs[11]);
	b    = mxGetPr(prhs[12]);
	Q    = mxGetPr(prhs[13]);
	Qf   = mxGetPr(prhs[14]);
	R    = mxGetPr(prhs[15]);
	S    = mxGetPr(prhs[16]);
	q    = mxGetPr(prhs[17]);
	qf   = mxGetPr(prhs[18]);
	r    = mxGetPr(prhs[19]);
	lb   = mxGetPr(prhs[20]);
	ub   = mxGetPr(prhs[21]);
	C    = mxGetPr(prhs[22]);
	D    = mxGetPr(prhs[23]);
	lg   = mxGetPr(prhs[24]);
	ug   = mxGetPr(prhs[25]);
	CN   = mxGetPr(prhs[26]);
	lgN  = mxGetPr(prhs[27]);
	ugN  = mxGetPr(prhs[28]);
	x    = mxGetPr(prhs[29]);
	u    = mxGetPr(prhs[30]);
	stat = mxGetPr(prhs[31]);

	int compute_res = (int) mxGetScalar(prhs[32]);
	inf_norm_res = mxGetPr(prhs[33]);
	
	int compute_mult = (int) mxGetScalar(prhs[34]);
	pi  = mxGetPr(prhs[35]);
	lam = mxGetPr(prhs[36]);
	t   = mxGetPr(prhs[37]);
	
	int kk = -1;

	int work_space_size = hpmpc_ip_hard_mpc_dp_work_space(N, nx, nu, nb, ng, ngN);
	
	double *work = (double *) malloc( work_space_size * sizeof(double) );

	// call solver 
	fortran_order_ip_hard_mpc(&kk, k_max, mu0, tol, 'd', N, nx, nu, nb, ng, ngN, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, C, D, lg, ug, CN, lgN, ugN, x, u, work, stat, compute_res, inf_norm_res, compute_mult, pi, lam, t);
	//c_order_ip_hard_mpc(&kk, k_max, mu0, tol, 'd', N, nx, nu, nb, ng, A, B, b, Q, Qf, S, R, q, qf, r, C, D, lb, ub, x, u, work, stat, compute_res, inf_norm_res, compute_mult, pi, lam, t);
	//c_order_ip_box_mpc(k_max, tol, 'd', nx, nu, N, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, x, u, work, &kk, stat);
	

	*kkk = (double) kk;

	free(work);

	return;

	}

