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

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_X86_ATOM) || defined(TARGET_AMD_SSE3)
#include <xmmintrin.h> // needed to flush to zero sub-normals with _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON); in the main()
#endif

// to throw floating-point exception
/*#define _GNU_SOURCE*/
/*#include <fenv.h>*/

#include "test_param.h"
#include "../problem_size.h"
#include "../include/aux_d.h"
#include "../include/lqcp_solvers.h"
#include "../include/mpc_solvers.h"
#include "../include/block_size.h"



int main()
	{
	
	printf("\n");
	printf("\n");
	printf("\n");
	printf(" HPMPC -- Library for High-Performance implementation of solvers for MPC.\n");
	printf(" Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.\n");
	printf("\n");
	printf(" HPMPC is distributed in the hope that it will be useful,\n");
	printf(" but WITHOUT ANY WARRANTY; without even the implied warranty of\n");
	printf(" MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
	printf(" See the GNU Lesser General Public License for more details.\n");
	printf("\n");
	printf("\n");
	printf("\n");
	
	int ii, jj;

	double **dummy;

	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line
	
	int nx, nu, N, nrep;

	nx = 25;
	nu = 1;
	N = 11;

	int rep;

	int nz = nx+nu+1;
	int anz = nal*((nz+nal-1)/nal);
	int anx = nal*((nx+nal-1)/nal);
	int pnz = bs*((nz+bs-1)/bs);
	int pnx = bs*((nx+bs-1)/bs);
	int pnu = bs*((nu+bs-1)/bs);
	int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	int cnx = ncl*((nx+ncl-1)/ncl);
	int cnu = ncl*((nu+ncl-1)/ncl);

	int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;

	const int ncx = nx;

/* matrices series */
	double *(hpL[N+1]);
	double *(hq[N+1]);
	double *(hux[N+1]);
	double *(hpi[N+1]);
	double *(hrb[N]);
	double *(hrq[N+1]);
	double *(hPb[N]);
	for(jj=0; jj<N; jj++)
		{
		d_zeros_align(&hq[jj], pnz, 1); // it has to be pnz !!!
		d_zeros_align(&hpL[jj], pnz, cnl);
		d_zeros_align(&hux[jj], pnz, 1); // it has to be pnz !!!
		d_zeros_align(&hpi[jj], pnx, 1);
		d_zeros_align(&hrb[jj], pnx, 1);
		d_zeros_align(&hrq[jj], pnz, 1);
		d_zeros_align(&hPb[jj], pnx, 1);
		}
	d_zeros_align(&hpL[N], pnz, cnl);
	d_zeros_align(&hq[N], pnz, 1); // it has to be pnz !!!
	d_zeros_align(&hux[N], pnz, 1); // it has to be pnz !!!
	d_zeros_align(&hpi[N], pnx, 1);
	d_zeros_align(&hrq[N], pnz, 1);


	double *diag; d_zeros_align(&diag, pnz, 1);
	
	//double *work; d_zeros_align(&work, 2*anz, 1);
	double *work; d_zeros_align(&work, pnz, cnx);

/************************************************
* test of riccati_eye / diag
************************************************/
	
#if 1
	float time_ip_diag, time_sv_diag, time_sv_full;

	#include "test_matrices_variable_nx.h"

	// horizon length
	N = 11;

	// base nx
	int nx0 = 2;
	int nu0 = 1;

	// size-varing
	int nxx[N+1];
	for(ii=0; ii<=N; ii++) nxx[ii] = (N+1-ii)*nx0 + nu0;

	int pnxx[N+1];
	for(ii=0; ii<=N; ii++) pnxx[ii] = (nxx[ii]+bs-1)/bs*bs;

	int cnxx[N+1];
	for(ii=0; ii<=N; ii++) cnxx[ii] = (nxx[ii]+ncl-1)/ncl*ncl;

	int nuu[N+1];
	for(ii=0; ii<N; ii++) nuu[ii] = nu0;
	nuu[N] = 0; // !!!!!

	int pnuu[N+1];
	for(ii=0; ii<N; ii++) pnuu[ii] = (nuu[ii]+bs-1)/bs*bs;
	pnuu[N] = 0; // !!!!!

	int cnuu[N+1];
	for(ii=0; ii<N; ii++) cnuu[ii] = (nuu[ii]+ncl-1)/ncl*ncl;
	cnuu[N] = 0; // !!!!!

	//for(ii=0; ii<=N; ii++) printf("\n%d %d %d\n", nxx[ii], pnxx[ii], cnxx[ii]);
	//for(ii=0; ii<N; ii++)  printf("\n%d %d %d\n", nuu[ii], pnuu[ii], cnuu[ii]);


	// data memory space
	double *(hdA[N]);
	double *(hpBt[N]);
	double *(hpR[N]);
	double *(hpS[N]);
	double *(hpQ2[N+1]);
	double *(hpLK[N]);
	double *(hpP[N+1]);
	double *pK;

	for(ii=0; ii<N; ii++)
		{
		d_zeros_align(&hdA[ii], pnxx[ii], 1);
		d_zeros_align(&hpBt[ii], pnuu[ii], cnxx[ii+1]);
		d_zeros_align(&hpR[ii], pnuu[ii], cnuu[ii]);
		d_zeros_align(&hpS[ii], pnxx[ii], cnuu[ii]);
		d_zeros_align(&hpQ2[ii], pnxx[ii], cnxx[ii]);
		d_zeros_align(&hpLK[ii], pnuu[ii]+pnxx[ii], cnuu[ii]);
		d_zeros_align(&hpP[ii], pnxx[ii], cnxx[ii]);
		}
	d_zeros_align(&hpQ2[N], pnxx[N], cnxx[N]);
	d_zeros_align(&hpP[N], pnxx[N], cnxx[N]);
	d_zeros_align(&pK, pnxx[0], cnuu[0]); // max(nx) x nax(nu)

	// A
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nxx[ii+1]; jj++)
			hdA[ii][jj] = 1.0;

	//d_print_mat(1, cnxx[1], hdA[0], 1);

	// B
	double *eye_nu0; d_zeros(&eye_nu0, nu0, nu0);
	for(jj=0; jj<nu0; jj++) eye_nu0[jj*(nu0+1)] = 1.0;
	double *ptrB = BBB;
	for(ii=0; ii<N; ii++)
		{
		d_cvt_mat2pmat(nuu[ii], nuu[ii], 0, bs, eye_nu0, nuu[ii], hpBt[ii], cnxx[ii]);
		d_cvt_tran_mat2pmat(nxx[ii+1]-nuu[ii], nuu[ii], 0, bs, ptrB, nxx[ii+1]-nuu[ii], hpBt[ii]+nuu[ii]*bs, cnxx[ii]);
		ptrB += nxx[ii+1] - nuu[ii];
		}
	free(eye_nu0);

	//d_print_pmat(pnuu[0], cnxx[1], bs, hpBt[0], cnxx[0]);
	//d_print_pmat(pnuu[1], cnxx[2], bs, hpBt[1], cnxx[1]);
	//d_print_pmat(pnuu[2], cnxx[3], bs, hpBt[2], cnxx[2]);
	//d_print_pmat(pnuu[N-1], cnxx[N-1], bs, hpBt[N-2], cnxx[N-2]);
	//d_print_pmat(pnuu[N-1], cnxx[N], bs, hpBt[N-1], cnxx[N-1]);

	// R
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nuu[ii]; jj++)
			hpR[ii][jj/bs*bs*cnuu[ii]+jj%bs+jj*bs] = 0.0;

	//for(ii=0; ii<N; ii++)
	//	d_print_pmat(pnuu[ii], cnuu[ii], bs, hpR[ii], pnuu[ii]);
	//d_print_pmat(pnuu[0], cnuu[0], bs, hpR[0], pnuu[0]);

	// S

	// Q
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<nu0; jj++)
			hpQ2[ii][jj/bs*bs*cnxx[ii]+jj%bs+jj*bs] = 1.0;
		for(jj=nxx[ii]-nx0; jj<nxx[ii]; jj++) 
			hpQ2[ii][jj/bs*bs*cnxx[ii]+jj%bs+jj*bs] = 1.0;
		}

	//for(ii=0; ii<=N; ii++)
	//	d_print_pmat(pnxx[ii], cnxx[ii], bs, hpQ2[ii], cnxx[ii]);
	//d_print_pmat(pnxx[0], cnxx[0], bs, hpQ2[0], cnxx[0]);
	//d_print_pmat(pnxx[1], cnxx[1], bs, hpQ2[1], cnxx[1]);
	//d_print_pmat(pnxx[N-1], cnxx[N-1], bs, hpQ2[N-1], cnxx[N-1]);
	//d_print_pmat(pnxx[N], cnxx[N], bs, hpQ2[N], cnxx[N]);
	//exit(1);


	double **ptr_dummy;


	printf("\nfactorize\n");
	d_ric_diag_trf_mpc(N, nxx, nuu, hdA, hpBt, hpR, hpS, hpQ2, hpLK, pK, hpP, diag, 0, ptr_dummy);
	printf("\nfactorize done\n");

#if 1
	//d_print_pmat(nxx[0], nxx[0], bs, hpP[0], cnxx[0]);
	//d_print_pmat(nxx[1], nxx[1], bs, hpP[1], cnxx[1]);
	//d_print_pmat(nxx[N-2], nxx[N-2], bs, hpP[N-2], cnxx[N-2]);
	//d_print_pmat(nxx[N-1], nxx[N-1], bs, hpP[N-1], cnxx[N-1]);
	//d_print_pmat(nxx[N], nxx[N], bs, hpP[N], cnxx[N]);

	//for(ii=0; ii<=N; ii++)
	//	d_print_pmat(pnuu[ii]+nxx[ii], nuu[ii], bs, hpLK[ii], cnuu[ii]);
	//d_print_pmat(pnuu[0]+nxx[0], nuu[0], bs, hpLK[0], cnuu[0]);
	//d_print_pmat(pnuu[1]+nxx[1], nuu[1], bs, hpLK[1], cnuu[1]);
	//d_print_pmat(pnuu[2]+nxx[2], nuu[2], bs, hpLK[2], cnuu[2]);
	//d_print_pmat(pnuu[N-3]+nxx[N-3], nuu[N-3], bs, hpLK[N-3], cnuu[N-3]);
	//d_print_pmat(pnuu[N-2]+nxx[N-2], nuu[N-2], bs, hpLK[N-2], cnuu[N-2]);
	//d_print_pmat(pnuu[N-1]+nxx[N-1], nuu[N-1], bs, hpLK[N-1], cnuu[N-1]);
#endif



	// data memory space
	double *(hrq2[N+1]);
	//double *(hu2[N]);
	//double *(hx2[N+1]);
	double *(hux2[N+1]);
	double *(hpi2[N+1]);
	double *(hPb2[N]);

	double *(hb2[N]);
	//double *(hres_r2[N]);
	//double *(hres_q2[N+1]);
	double *(hres_rq2[N+1]);
	double *(hres_b2[N]);

	for(ii=0; ii<N; ii++)
		{
		d_zeros_align(&hrq2[ii], pnuu[ii]+pnxx[ii], 1);
		//d_zeros_align(&hu2[ii], pnuu[ii], 1);
		//d_zeros_align(&hx2[ii], pnxx[ii], 1);
		d_zeros_align(&hux2[ii], pnuu[ii]+pnxx[ii], 1);
		d_zeros_align(&hpi2[ii], pnxx[ii], 1);
		d_zeros_align(&hPb2[ii], pnxx[ii+1], 1);

		d_zeros_align(&hb2[ii], pnxx[ii+1], 1);
		//d_zeros_align(&hres_r2[ii], pnuu[ii], 1);
		//d_zeros_align(&hres_q2[ii], pnxx[ii], 1);
		d_zeros_align(&hres_rq2[ii], pnuu[ii]+pnxx[ii], 1);
		d_zeros_align(&hres_b2[ii], pnxx[ii+1], 1);
		}
	d_zeros_align(&hrq2[N], pnuu[N]+pnxx[N], 1);
	//d_zeros_align(&hx2[N], pnxx[N], 1);
	d_zeros_align(&hux2[N], pnuu[N]+pnxx[N], 1);
	d_zeros_align(&hpi2[N], pnxx[N], 1);

	d_zeros_align(&hres_rq2[N], pnuu[N]+pnxx[N], 1);

	double *work_diag2; d_zeros_align(&work_diag2, pnxx[0], 1);

	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nuu[ii]; jj++)
			hrq2[ii][jj] = 0.0;

	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nxx[ii]; jj++)
			hrq2[ii][nuu[ii]+jj] = 0.0;

	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nxx[ii+1]; jj++)
			hb2[ii][jj] = 0.0;

	// x0
	//hux2[0][2*nuu[0]+0] =  5.0;
	//hux2[0][2*nuu[0]+1] = -5.0;
	for(jj=0; jj<nuu[0]; jj++)
		{
		hux2[0][jj] = 0.0;
		}
	for(; jj<nuu[0]+nu; jj++)
		{
		hux2[0][jj] = 7.5097;
		}
	for(; jj<nxx[0]; jj+=2)
		{
		hux2[0][jj+0] = 15.01940;
		hux2[0][jj+1] =  0.0;
		}
	d_print_mat(1, nuu[0]+nxx[0], hux2[0], 1);


	printf("\nsolve\n");
	d_ric_diag_trs_mpc(N, nxx, nuu, hdA, hpBt, hpLK, hpP, hb2, hrq2, hux2, 1, hPb2, 1, hpi2, work_diag2);
	printf("\nsolve done\n");

#if 1
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nuu[ii]+nxx[ii], hux2[ii], 1);
#endif



	// residuals

#if 1
	printf("\nresudyals\n");
	d_res_diag_mpc(N, nxx, nuu, hdA, hpBt, hpR, hpS, hpQ2, hb2, hrq2, hux2, hpi2, hres_rq2, hres_b2, work_diag2);
	printf("\nresiduals done\n");

	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nuu[ii]+nxx[ii], hres_rq2[ii], 1);

	for(ii=0; ii<N; ii++)
		d_print_mat(1, nxx[ii+1], hres_b2[ii], 1);
#endif





	// timing
	struct timeval tv20, tv21;

#if 1
	gettimeofday(&tv20, NULL); // start

	nrep = 10000;
	for(ii=0; ii<nrep; ii++)
		{
		d_ric_diag_trf_mpc(N, nxx, nuu, hdA, hpBt, hpR, hpS, hpQ2, hpLK, pK, hpP, diag, 0, ptr_dummy);
		d_ric_diag_trs_mpc(N, nxx, nuu, hdA, hpBt, hpLK, hpP, hb2, hrq2, hux2, 1, hPb2, 1, hpi2, work_diag2);
		}

	gettimeofday(&tv21, NULL); // start

	time_sv_diag = (float) (tv21.tv_sec-tv20.tv_sec)/(nrep+0.0)+(tv21.tv_usec-tv20.tv_usec)/(nrep*1e6);
#endif





#if 1
	// IPM
	int kk = -1;
	int kmax = 50;
	double mu0 = 1;
	double mu_tol = 1e-8;
	double alpha_min = 1e-12;
	double sigma_par[] = {0.4, 0.3, 0.01};
	double stat[5*50] = {};

	int nbb[N+1];
	for(ii=0; ii<=N; ii++) nbb[ii] = nuu[ii] + nxx[ii];
	nbb[0] = nuu[0]; // XXX !!!!!!!!!!!!!!

	int pnbb[N+1];
	for(ii=0; ii<=N; ii++) pnbb[ii] = (nbb[ii]+bs-1)/bs*bs;

	double *(hd2[N+1]);
	double *(hlam2[N+1]);
	double *(ht2[N+1]);
	double *(hres_d2[N+1]);
	for(ii=0; ii<=N; ii++)
		{
		d_zeros_align(&hd2[ii], 2*pnbb[ii], 1);
		d_zeros_align(&hlam2[ii], 2*pnbb[ii], 1);
		d_zeros_align(&ht2[ii], 2*pnbb[ii], 1);
		d_zeros_align(&hres_d2[ii], 2*pnbb[ii], 1);
		}

	double mu2 = -1;

	printf("\nbounds\n");
	ii = 0; // initial stage
	for(jj=0; jj<nuu[ii]; jj++)
		{
		hd2[ii][jj]                  = -20.5;
		hd2[ii][pnbb[ii]+jj]         = -20.5;
		}
	d_print_mat(1, 2*pnbb[ii], hd2[ii], 1);
	for(ii=1; ii<=N; ii++)
		{
		for(jj=0; jj<nuu[ii]; jj++)
			{
			hd2[ii][jj]          = -20.5;
			hd2[ii][pnbb[ii]+jj] = -20.5;
			}
		for(; jj<nuu[ii]+nu0; jj++)
			{
			hd2[ii][jj]          = - 2.5;
			hd2[ii][pnbb[ii]+jj] = -10.0;
			}
		for(; jj<nbb[ii]-nx0; jj++)
		//for(; jj<nbb[ii]; jj++)
			{
			hd2[ii][jj]          = -100.0;
			hd2[ii][pnbb[ii]+jj] = -100.0;
			}
		hd2[ii][jj+0]          = - 0.0; //   0
		hd2[ii][pnbb[ii]+jj+0] = -20.0; // -20
		hd2[ii][jj+1]          = -10.0; // -10
		hd2[ii][pnbb[ii]+jj+1] = -10.0; // -10
		d_print_mat(1, 2*pnbb[ii], hd2[ii], 1);
		}

	for(jj=0; jj<nuu[0]; jj++)
		{
		hux2[0][jj] = 0.0;
		}
	for(; jj<nuu[0]+nu; jj++)
		{
		hux2[0][jj] = 7.5097;
		}
	for(; jj<nxx[0]; jj+=2)
		{
		hux2[0][jj+0] = 15.01940;
		hux2[0][jj+1] =  0.0;
		}
	d_print_mat(1, nuu[0]+nxx[0], hux2[0], 1);


	int pnxM = pnxx[0];
	int pnuM = pnuu[0];
	int cnuM = cnuu[0];

	int anxx[N+1];
	for(ii=0; ii<=N; ii++) anxx[ii] = (nxx[ii]+nal-1)/nal*nal;

	int anuu[N+1];
	for(ii=0; ii<=N; ii++) anuu[ii] = (nuu[ii]+nal-1)/nal*nal;

	int work_space_ip_double = 0;
	for(ii=0; ii<=N; ii++)
		work_space_ip_double += anuu[ii] + 3*anxx[ii] + (pnuu[ii]+pnxx[ii])*cnuu[ii] + pnxx[ii]*cnxx[ii] + 3*pnxx[ii] + 3*pnuu[ii] + 8*pnbb[ii];
	work_space_ip_double += pnxM*cnuM + pnxM + pnuM;
	int work_space_ip_int = (N+1)*7*sizeof(int);
	work_space_ip_int = (work_space_ip_int+63)/64*64;
	work_space_ip_int /= sizeof(int);
	printf("\nwork space: %d double + %d int\n", work_space_ip_double, work_space_ip_int);
	double *work_space_ip; d_zeros_align(&work_space_ip, work_space_ip_double+(work_space_ip_int+1)/2, 1); // XXX assume sizeof(double) = 2 * sizeof(int) !!!!!


	printf("\nIPM\n");
	d_ip2_diag_mpc(&kk, kmax, mu0, mu_tol, alpha_min, 0, sigma_par, stat, N, nxx, nuu, nbb, hdA, hpBt, hpR, hpS, hpQ2, hb2, hd2, hrq2, hux2, 1, hpi2, hlam2, ht2, work_space_ip);
	printf("\nIPM done\n");


	printf("\nux\n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nuu[ii]+nxx[ii], hux2[ii], 1);

	printf("\nlam\n");
	for(ii=0; ii<=N; ii++)
		{
		d_print_mat(1, nbb[ii], hlam2[ii], 1);
		d_print_mat(1, nbb[ii], hlam2[ii]+pnbb[ii], 1);
		}

	printf("\nt\n");
	for(ii=0; ii<=N; ii++)
		{
		d_print_mat(1, nbb[ii], ht2[ii], 1);
		d_print_mat(1, nbb[ii], ht2[ii]+pnbb[ii], 1);
		}

	printf("\nstat\n\n");
	for(ii=0; ii<kk; ii++)
		printf("%f %f %f %f %f\n", stat[5*ii+0], stat[5*ii+1], stat[5*ii+2], stat[5*ii+3], stat[5*ii+4]);
	printf("\n\n");


	// residuals
	printf("\nresuduals IPM\n");
	d_res_ip_diag_mpc(N, nxx, nuu, nbb, hdA, hpBt, hpR, hpS, hpQ2, hb2, hrq2, hd2, hux2, hpi2, hlam2, ht2, hres_rq2, hres_b2, hres_d2, &mu2, work_diag2);
	printf("\nresiduals IPM done\n");

	printf("\nres_rq\n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nuu[ii]+nxx[ii], hres_rq2[ii], 1);

	printf("\nres_b\n");
	for(ii=0; ii<N; ii++)
		d_print_mat(1, nxx[ii+1], hres_b2[ii], 1);

	printf("\nres_d\n");
	for(ii=0; ii<=N; ii++)
		{
		d_print_mat(1, nbb[ii], hres_d2[ii], 1);
		d_print_mat(1, nbb[ii], hres_d2[ii]+pnbb[ii], 1);
		}

	printf("\nres_mu\n");
	d_print_mat(1, 1, &mu2, 1);


	// timing
	gettimeofday(&tv20, NULL); // start

	for(ii=0; ii<nrep; ii++)
		{
		d_ip2_diag_mpc(&kk, kmax, mu0, mu_tol, alpha_min, 0, sigma_par, stat, N, nxx, nuu, nbb, hdA, hpBt, hpR, hpS, hpQ2, hb2, hd2, hrq2, hux2, 1, hpi2, hlam2, ht2, work_space_ip);
		}

	gettimeofday(&tv21, NULL); // start

	time_ip_diag = (float) (tv21.tv_sec-tv20.tv_sec)/(nrep+0.0)+(tv21.tv_usec-tv20.tv_usec)/(nrep*1e6);


	for(ii=0; ii<=N; ii++)
		{
		free(hd2[ii]);
		free(hlam2[ii]);
		free(ht2[ii]);
		}
	free(work_space_ip);
#endif


	for(ii=0; ii<N; ii++)
		{
		free(hdA[ii]);
		free(hpBt[ii]);
		free(hpR[ii]);
		free(hpS[ii]);
		free(hpQ2[ii]);
		free(hpLK[ii]);
		free(hpP[ii]);
		free(hrq2[ii]);
		//free(hx2[ii]);
		//free(hu2[ii]);
		free(hux2[ii]);
		free(hpi2[ii]);
		free(hPb2[ii]);

		free(hb2[ii]);
		//free(hres_r2[ii]);
		//free(hres_q2[ii]);
		free(hres_rq2[ii]);
		free(hres_b2[ii]);
		}
	free(hpQ2[N]);
	free(hpP[N]);
	free(pK);
	free(hrq2[N]);
	//free(hx2[N]);
	free(hux2[N]);
	free(hpi2[N]);
	free(work_diag2);

	free(hres_rq2[N]);






	// reference using time-invariant code
	nx = 25;
	nu = 1;
	N  = 11;

	nz  = nx+nu+1;
	pnx = (nx+bs-1)/bs*bs;
	pnz = (nx+nu+1+bs-1)/bs*bs;
	cnx = (nx+ncl-1)/ncl*ncl;
	cnz = (nz+ncl-1)/ncl*ncl;

	double *BAb_temp; d_zeros(&BAb_temp, nx, nu+nx+1);
	double *(hpBAbt2[N]);

	ptrB = BBB;
	for(ii=0; ii<N; ii++)
		{
		//printf("\n%d\n", ii);
		d_zeros_align(&hpBAbt2[ii], pnz, cnx);
		for(jj=0; jj<nx*(nx+nu+1); jj++) BAb_temp[jj] = 0.0;
		for(jj=0; jj<nu; jj++) BAb_temp[jj*(nx+1)] = 1.0;
		d_copy_mat(nxx[ii+1]-1, nuu[ii], ptrB, nxx[ii+1]-1, BAb_temp+1, nx);
		ptrB += nxx[ii+1]-1;
		for(jj=0; jj<nxx[ii+1]; jj++) BAb_temp[nuu[ii]*nx+jj*(nx+1)] = 1.0;
		//for(jj=0; jj<nxx[ii+1]; jj++) BAb_temp[(nuu[ii]+nxx[ii+1])*nx+jj] = 1.0;
		//d_print_mat(nx, nu+nx+1, BAb_temp, nx);
		d_cvt_tran_mat2pmat(nx, nx+nu+1, 0, bs, BAb_temp, nx, hpBAbt2[ii], cnx);
		//d_print_pmat(nx+nu+1, nx, bs, hpBAbt2[ii], cnx);
		}

	double *RSQ; d_zeros(&RSQ, nz, nz);
	double *(hpRSQ[N+1]);

	for(ii=0; ii<=N; ii++)
		{
		//printf("\n%d\n", ii);
		d_zeros_align(&hpRSQ[ii], pnz, cnz);
		for(jj=0; jj<nz*nz; jj++) RSQ[jj] = 0.0;
		for(jj=nu; jj<2*nu; jj++) RSQ[jj*(nz+1)] = 1.0;
		for(jj=nu+nxx[ii]-nx0; jj<nu+nxx[ii]; jj++) RSQ[jj*(nz+1)] = 1.0;
		d_cvt_mat2pmat(nz, nz, 0, bs, RSQ, nz, hpRSQ[ii], cnz);
		//d_print_pmat(nz, nz, bs, hpRSQ[ii], cnz);
		}

	for(jj=0; jj<nx+nu; jj++) hux[0][jj] = 0.0;
	//hux[0][2*nu+0] =  5.0;
	//hux[0][2*nu+1] = -5.0;
	for(jj=0; jj<nu; jj++)
		{
		hux[0][nu+jj] = 7.5097;
		}
	for(; jj<nx; jj+=2)
		{
		hux[0][nu+jj+0] = 15.01940;
		hux[0][nu+jj+1] =  0.0;
		}

	d_ric_sv_mpc(nx, nu, N, hpBAbt2, hpRSQ, 0, dummy, dummy, hux, hpL, work, diag, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy, 0);
	//for(ii=0; ii<=N; ii++)
	//	d_print_pmat(pnz, cnl-3, bs, hpL[ii], cnl);
	//d_print_pmat(pnz, nu, bs, hpL[0], cnl);
	//d_print_pmat(pnz, cnl-3, bs, hpL[1], cnl);
	//d_print_pmat(pnz, cnl-3, bs, hpL[2], cnl);
	//d_print_pmat(pnz, cnl-3, bs, hpL[N-3], cnl);
	//d_print_pmat(pnz, cnl-3, bs, hpL[N-2], cnl);
	//d_print_pmat(pnz, cnl-3, bs, hpL[N-1], cnl);
	//d_print_pmat(pnz, cnl, bs, hpL[N], cnl);
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nx+nu, hux[ii], 1);

	
	double *(hq3[N+1]);
	double *(hrq3[N+1]);
	double *(hrb3[N]);

	for(ii=0; ii<N; ii++)
		{
		d_zeros_align(&hq3[ii], pnz, 1);
		d_zeros_align(&hrq3[ii], pnz, 1);
		d_zeros_align(&hrb3[ii], pnx, 1);
		}
	d_zeros_align(&hq3[N], pnz, 1);
	d_zeros_align(&hrq3[N], pnz, 1);
	

	d_res_mpc(nx, nu, N, hpBAbt2, hpRSQ, hq3, hux, hpi, hrq3, hrb3);

	printf("\nresiduals\n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nx+nu, hrq3[ii], 1);

	for(ii=0; ii<N; ii++)
		d_print_mat(1, nx, hrb3[ii], 1);


	for(ii=0; ii<N; ii++)
		{
		free(hq3[ii]);
		free(hrq3[ii]);
		free(hrb3[ii]);
		}
	free(hq3[N]);
	free(hrq3[N]);


	// timing
	//struct timeval tv20, tv21;

#if 1
	gettimeofday(&tv20, NULL); // start

	for(ii=0; ii<nrep; ii++)
		{
		d_ric_sv_mpc(nx, nu, N, hpBAbt2, hpRSQ, 0, dummy, dummy, hux, hpL, work, diag, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy, 0);
		}

	gettimeofday(&tv21, NULL); // start

	time_sv_full = (float) (tv21.tv_sec-tv20.tv_sec)/(nrep+0.0)+(tv21.tv_usec-tv20.tv_usec)/(nrep*1e6);

	printf("\ndiag time = %e\t\tfull time = %e\t\tip diag time = %e\n\n", time_sv_diag, time_sv_full, time_ip_diag);
#endif



	free(BAb_temp);
	for(ii=0; ii<N; ii++)
		{
		free(hpBAbt2[ii]);
		free(hpRSQ[ii]);
		}
	free(hpRSQ[N]);

	exit(1);


#endif

	}
