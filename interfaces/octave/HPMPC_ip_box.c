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
#include <math.h>

#include "hpmpc/aux_d.h"
/*#include "hpmpc/block_size.h"*/
#include "hpmpc/mpc_solvers.h"



#define IP 1




/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
	{
		
	/* get data */
	int k_max, nx, nu, N, nb;
	double tol, *A, *B, *b, *Q, *Qf, *R, *S, *q, *qf, *r, *x, *u, *lb, *ub, *stat, *kkk;
	
	k_max = (int) mxGetScalar(prhs[0]);
	tol = mxGetScalar(prhs[1]);
	nx = (int) mxGetScalar(prhs[2]);
	nu = (int) mxGetScalar(prhs[3]);
	N  = (int) mxGetScalar(prhs[4]);
	nb  = (int) mxGetScalar(prhs[5]);

	A = mxGetPr(prhs[6]);
	B = mxGetPr(prhs[7]);
	b = mxGetPr(prhs[8]);
	Q = mxGetPr(prhs[9]);
	Qf = mxGetPr(prhs[10]);
	R = mxGetPr(prhs[11]);
	S = mxGetPr(prhs[12]);
	q = mxGetPr(prhs[13]);
	qf = mxGetPr(prhs[14]);
	r = mxGetPr(prhs[15]);
	x = mxGetPr(prhs[16]);
	u = mxGetPr(prhs[17]);
	lb = mxGetPr(prhs[18]);
	ub = mxGetPr(prhs[19]);
	stat = mxGetPr(prhs[20]);
	kkk = mxGetPr(prhs[21]);

	/* utility */
	int ii, jj;
	const int bsd = d_get_mr();
	int nz = nx+nu+1;
	int pnz = bsd*((nz+bsd-nu%bsd+bsd-1)/bsd);
	
	/* pack */
	double *BAbt; d_zeros(&BAbt, pnz, nx);
	for(ii=0; ii<nx; ii++)
		for(jj=0; jj<nu; jj++)
			BAbt[jj+pnz*ii] = B[ii+nx*jj];
	for(ii=0; ii<nx; ii++)
		for(jj=0; jj<nx; jj++)
			BAbt[nu+jj+pnz*ii] = A[ii+nx*jj];
	for(ii=0; ii<nx; ii++)
		BAbt[nu+nx+pnz*ii] = b[ii];
	
    /* d_print_mat(nz, nx, BAbt, pnz); */

	double *P; d_zeros_align(&P, pnz, pnz);
	for(jj=0; jj<nu; jj++)
		for(ii=0; ii<nu; ii++)
			P[ii+pnz*jj] = R[ii+nu*jj];
	for(jj=0; jj<nx; jj++)
		for(ii=0; ii<nx; ii++)
			P[nu+ii+pnz*(nu+jj)] = Q[ii+nx*jj];
	for(ii=0; ii<nu; ii++)
		P[nu+nx+pnz*ii] = r[ii];
	for(ii=0; ii<nx; ii++)
		P[nu+nx+pnz*(nu+ii)] = q[ii];
	P[(nu+nx)*(pnz+1)] = 1e6;

   /*	d_print_mat(nx, nx, Q, nx); */
   /*	d_print_mat(nz, nz, P, pnz); */

	double *Pf; d_zeros_align(&Pf, pnz, pnz);
	for(ii=0; ii<nu; ii++)
		Pf[ii*(pnz+1)] = 1.0;
	for(jj=0; jj<nx; jj++)
		for(ii=0; ii<nx; ii++)
			Pf[nu+ii+pnz*(nu+jj)] = Qf[ii+nx*jj];
	for(ii=0; ii<nx; ii++)
		Pf[nu+nx+pnz*(nu+ii)] = qf[ii];
	Pf[(nu+nx)*(pnz+1)] = 1e6;

    /*	d_print_mat(nz, nz, Pf, pnz); */

	/* block */
	double *pBAbt; d_zeros_align(&pBAbt, pnz, pnz);
	d_cvt_mat2pmat(nz, nx, 0, bsd, BAbt, pnz, pBAbt, pnz);

	double *pQ; d_zeros_align(&pQ, pnz, pnz);
	d_cvt_mat2pmat(nz, nz, 0, bsd, P, pnz, pQ, pnz);

	double *pQf; d_zeros_align(&pQf, pnz, pnz);
	d_cvt_mat2pmat(nz, nz, 0, bsd, Pf, pnz, pQf, pnz);

    /*	d_print_pmat(nz, nx, bsd, pBAbt, pnz); */
    /*	d_print_pmat(nz, nz, bsd, pQ, pnz); */

	/* matrices series */
	double *ptr_Q; d_zeros_align(&ptr_Q, pnz*pnz, N+1);
	double *ptr_ux; d_zeros_align(&ptr_ux, pnz, N+1);
	double *ptr_pi; d_zeros_align(&ptr_pi, nx, N+1);
	double *ptr_t; d_zeros_align(&ptr_t, 2*nb, N+1);
	double *ptr_lam; d_zeros_align(&ptr_lam, 2*nb, N+1);
	double *(hpBAbt[N]);
	double *(hpQ[N+1]);
	double *(hux[N+1]);
	double *(hpi[N+1]);
	double *(hlam[N+1]);
	double *(ht[N+1]);
	double *(hlb[N+1]);
	double *(hub[N+1]);
	for(ii=0; ii<N; ii++)
		{
		hpBAbt[ii] = pBAbt;
		hpQ[ii] = ptr_Q + ii*pnz*pnz;
		hux[ii] = ptr_ux + ii*pnz;
		hpi[ii] = ptr_pi + ii*nx;
		ht[ii] = ptr_t + ii*2*nb;
		hlam[ii] = ptr_lam + ii*2*nb;
		hlb[ii] = lb;
		hub[ii] = ub;
		}
	hpQ[ii] = ptr_Q + N*pnz*pnz;
	hux[N] = ptr_ux + N*pnz;
	hpi[N] = ptr_pi + N*nx;
	ht[N] = ptr_t + N*2*nb;
	hlam[N] = ptr_lam + N*2*nb;
	hlb[N] = lb;
	hub[N] = ub;
	
	/* copy cost function */
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<pnz*pnz; jj++) hpQ[ii][jj]=pQ[jj];
	for(jj=0; jj<pnz*pnz; jj++) hpQ[N][jj]=pQf[jj];

	/* work space */
/*	double *work; d_zeros_align(&work, (N+1)*(pnz*pnz+pnz+5*nb)+2*pnz*pnz, 1); // work space*/
	double *work; d_zeros_align(&work, 2*((N+1)*(pnz*pnz+2*pnz+2*4*nb)+2*pnz*pnz), 1); // work space
	
	/* ip stuff */
	int kk=-1;
	double sigma[] = {0.4, 0.3, 0.01};      /* control primal-dual IP behaviour */
	int warm_start = 0;
	int info = 0;
	int compute_mult = 0;
//	double *info; d_zeros(&info, 5, k_max); /* infos from the IP routine */

	/* copy initial guess */
	for(jj=0; jj<N; jj++)
		for(ii=0; ii<nu; ii++)
			hux[jj][ii] = u[ii+nu*jj];
	for(jj=0; jj<=N; jj++)
		for(ii=0; ii<nx; ii++)
			hux[jj][nu+ii] = x[ii+nx*jj];

	/* call the solver */
/*	ip_d_box(&kk, k_max, tol, sigma, info, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, work);*/
	if(IP==1)
		d_ip_box(&kk, k_max, tol, warm_start, sigma, stat, nx, nu, N, nb, hpBAbt, hpQ, hlb, hub, hux, compute_mult, hpi, hlam, ht, work, &info);
	else
		d_ip2_box(&kk, k_max, tol, warm_start, sigma, stat, nx, nu, N, nb, hpBAbt, hpQ, hlb, hub, hux, compute_mult, hpi, hlam, ht, work, &info);
	
	if(info!=0)
		{
		printf("\nError: Hessian not positive definite.\n");
		}
	else
		{
		printf("\nSolution OK.\n");
		}
	
	*kkk = (double) kk;

	/* copy back solution */
	for(jj=0; jj<N; jj++)
		for(ii=0; ii<nu; ii++)
			u[ii+nu*jj] = hux[jj][ii];
	for(jj=1; jj<=N; jj++)
		for(ii=0; ii<nx; ii++)
			x[ii+nx*jj] = hux[jj][nu+ii];
			
/*	printf("\nu = \n\n");*/
/*	for(ii=0; ii<N; ii++)*/
/*		d_print_mat(1, nu, hux[ii], 1);*/

/*	printf("\nX = \n\n");*/
/*	for(ii=0; ii<=N; ii++)*/
/*		d_print_mat(1, nx, hux[ii]+nu, 1);*/

	/* free memory */
	free(BAbt);
	free(P);
	free(Pf);
	free(pBAbt);
	free(pQ);
	free(pQf);
	free(ptr_Q);
	free(ptr_ux);
/*printf("\nciao\n");*/
	free(ptr_pi);
	free(ptr_t);
	free(ptr_lam);
	free(work);
//	free(info);
	
	/* return; */
	return;

	}

