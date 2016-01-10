/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.                *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                                                                                                 *
**************************************************************************************************/

#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
/*#include <math.h>*/

// static or dynamic memory allocation
#define STATIC_MEMORY 0

// definition of problem sizes needed for static memory allocation
#if STATIC_MEMORY
	#define NX 12
	#define NU 5
	#define NN 30
	#define NB (NU+NX)
	#define NG 0
	#define NGN NX
#endif

// include macro for work space size
#include "/opt/hpmpc/include/c_interface.h"



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
	const int time_invariant = (int) mxGetScalar(prhs[10]);

	A    = mxGetPr(prhs[11]);
	B    = mxGetPr(prhs[12]);
	b    = mxGetPr(prhs[13]);
	Q    = mxGetPr(prhs[14]);
	Qf   = mxGetPr(prhs[15]);
	R    = mxGetPr(prhs[16]);
	S    = mxGetPr(prhs[17]);
	q    = mxGetPr(prhs[18]);
	qf   = mxGetPr(prhs[19]);
	r    = mxGetPr(prhs[20]);
	lb   = mxGetPr(prhs[21]);
	ub   = mxGetPr(prhs[22]);
	C    = mxGetPr(prhs[23]);
	D    = mxGetPr(prhs[24]);
	lg   = mxGetPr(prhs[25]);
	ug   = mxGetPr(prhs[26]);
	CN   = mxGetPr(prhs[27]);
	lgN  = mxGetPr(prhs[28]);
	ugN  = mxGetPr(prhs[29]);
	x    = mxGetPr(prhs[30]);
	u    = mxGetPr(prhs[31]);
	stat = mxGetPr(prhs[32]);

	int compute_res = (int) mxGetScalar(prhs[33]);
	inf_norm_res = mxGetPr(prhs[34]);
	
	int compute_mult = (int) mxGetScalar(prhs[35]);
	pi  = mxGetPr(prhs[36]);
	lam = mxGetPr(prhs[37]);
	t   = mxGetPr(prhs[38]);
	
	int kk = -1;

#if (STATIC_MEMORY==1)
	static double work[HPMPC_IP_MPC_DP_WORK_SPACE_TV];
#else
	int work_space_size = hpmpc_ip_hard_mpc_dp_work_space_tv(N, nx, nu, nb, ng, ngN);
	double *work = (double *) malloc( work_space_size * sizeof(double) );
#endif

	// call solver 
	fortran_order_ip_hard_mpc_tv(&kk, k_max, mu0, tol, 'd', N, nx, nu, nb, ng, ngN, time_invariant, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, C, D, lg, ug, CN, lgN, ugN, x, u, work, stat, compute_res, inf_norm_res, compute_mult, pi, lam, t);
	//c_order_ip_hard_mpc_tv(&kk, k_max, mu0, tol, 'd', N, nx, nu, nb, ng, ngN, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, C, D, lg, ug, CN, lgN, ugN, x, u, work, stat, compute_res, inf_norm_res, compute_mult, pi, lam, t);

	*kkk = (double) kk;

#if (STATIC_MEMORY!=1)
	free(work);
#endif

	return;

	}

