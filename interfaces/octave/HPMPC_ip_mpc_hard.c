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
	const int free_x0 = (int) mxGetScalar(prhs[11]);
	const int warm_start = (int) mxGetScalar(prhs[12]);

	A    = mxGetPr(prhs[13]);
	B    = mxGetPr(prhs[14]);
	b    = mxGetPr(prhs[15]);
	Q    = mxGetPr(prhs[16]);
	Qf   = mxGetPr(prhs[17]);
	R    = mxGetPr(prhs[18]);
	S    = mxGetPr(prhs[19]);
	q    = mxGetPr(prhs[20]);
	qf   = mxGetPr(prhs[21]);
	r    = mxGetPr(prhs[22]);
	lb   = mxGetPr(prhs[23]);
	ub   = mxGetPr(prhs[24]);
	C    = mxGetPr(prhs[25]);
	D    = mxGetPr(prhs[26]);
	lg   = mxGetPr(prhs[27]);
	ug   = mxGetPr(prhs[28]);
	CN   = mxGetPr(prhs[29]);
	lgN  = mxGetPr(prhs[30]);
	ugN  = mxGetPr(prhs[31]);
	x    = mxGetPr(prhs[32]);
	u    = mxGetPr(prhs[33]);
	pi  = mxGetPr(prhs[34]);
	lam = mxGetPr(prhs[35]);
	t   = mxGetPr(prhs[36]);
	inf_norm_res = mxGetPr(prhs[37]);
	stat = mxGetPr(prhs[38]);
	
	int kk = -1;

#if (STATIC_MEMORY==1)
	static double work[HPMPC_IP_MPC_DP_WORK_SPACE_TV];
#else
	int work_space_size = hpmpc_d_ip_mpc_hard_tv_work_space_size_doubles(N, nx, nu, nb, ng, ngN);
	double *work = (double *) malloc( work_space_size * sizeof(double) );
#endif

	// call solver 
	fortran_order_d_ip_mpc_hard_tv(&kk, k_max, mu0, tol, N, nx, nu, nb, ng, ngN, time_invariant, free_x0, warm_start, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, C, D, lg, ug, CN, lgN, ugN, x, u, pi, lam, t, inf_norm_res, work, stat);
//	c_order_d_ip_mpc_hard_tv(&kk, k_max, mu0, tol, N, nx, nu, nb, ng, ngN, time_invariant, free_x0, warm_start, A, B, b, Q, Qf, S, R, q, qf, r, lb, ub, C, D, lg, ug, CN, lgN, ugN, x, u, pi, lam, t, inf_norm_res, work, stat);

	*kkk = (double) kk;

#if (STATIC_MEMORY!=1)
	free(work);
#endif

	return;

	}

