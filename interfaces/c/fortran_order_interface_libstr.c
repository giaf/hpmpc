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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef BLASFEO
#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_blas.h>
#include <blasfeo_d_aux.h>
#endif

#include "../../include/target.h"
#include "../../include/block_size.h"
#include "../../include/aux_d.h"
#include "../../include/aux_s.h"
#include "../../include/blas_d.h"
#include "../../include/lqcp_solvers.h"
#include "../../include/mpc_solvers.h"

// Debug flag
#ifndef PC_DEBUG
#define PC_DEBUG 0
#endif /* PC_DEBUG */



// TODO partial condensing
int hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int N2)
	{
	int ii;
	int size = 0;
	for(ii=0; ii<N; ii++)
		{
		size += d_size_strmat(nu[ii]+nx[ii]+1, nx[ii+1]); // BAbt
		size += d_size_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii]); // RSQrq
		size += d_size_strmat(nu[ii]+nx[ii]+1, ng[ii]); // DCt
		size += 3*d_size_strvec(nx[ii]); // b, rb, pi
		size += 3*d_size_strvec(nu[ii]+nx[ii]); // rq, rrq, ux
		size += 5*d_size_strvec(2*nb[ii]+2*ng[ii]); // d, lam, t, rd, rm
		}
	size += d_ip2_res_mpc_hard_work_space_size_bytes_libstr(N, nx, nu, nb, ng);
	size += 64; // typical cache line size, used for alignement
	return size;
	}



// TODO partial condensing
int fortran_order_d_ip_ocp_hard_tv( 
							int *kk, int k_max, double mu0, double mu_tol,
							int N, int *nx, int *nu, int *nb, int **hidxb, int *ng,
							int N2,
							int warm_start,
							double **A, double **B, double **b, 
							double **Q, double **S, double **R, double **q, double **r, 
							double **lb, double **ub,
							double **C, double **D, double **lg, double **ug,
							double **x, double **u, double **pi, double **lam, //double **t,
							double *inf_norm_res,
							void *work0, 
							double *stat)

	{

//printf("\nstart of wrapper\n");

	int hpmpc_status = -1;


	int ii, jj, ll, idx;



	// XXX sequential update not implemented
	if(N2>N)
		N2 = N;



	// if ng>0, disable partial condensing (TODO implement this case)
	int ngM = ng[0];
	for(ii=1; ii<N; ii++)
		ngM = ng[ii]> ngM ? ng[ii] : ngM;
	
	if(ngM>0)
		N2 = N;



	// check for consistency of problem size
	// nb <= nu+nx
	for(ii=0; ii<=N; ii++)
		{
		if(nb[ii]>nu[ii]+nx[ii])
			{
			printf("\nERROR: At stage %d, the number of bounds nb=%d can not be larger than the number of variables nu+nx=%d.\n\n", ii, nb[ii], nu[ii]+nx[ii]);
			exit(1);
			}
		}


	double alpha_min = 1e-8; // minimum accepted step length
	double temp;
	
	int info = 0;




//printf("\n%d\n", ((size_t) work0) & 63);

	// align to (typical) cache line size
	size_t addr = (( (size_t) work0 ) + 63 ) / 64 * 64;
	char *c_ptr = (char *) addr;


//printf("\n%d\n", ((size_t) ptr) & 63);

	// data structure
	struct d_strmat hsBAbt[N];
	struct d_strmat hsRSQrq[N+1];
	struct d_strmat hsDCt[N+1];
	struct d_strvec hsb[N];
	struct d_strvec hsrq[N+1];
	struct d_strvec hsd[N+1];
	struct d_strvec hsux[N+1];
	struct d_strvec hspi[N+1];
	struct d_strvec hslam[N+1];
	struct d_strvec hst[N+1];
	struct d_strvec hsrrq[N+1];
	struct d_strvec hsrb[N];
	struct d_strvec hsrd[N+1];
	struct d_strvec hsrm[N+1];
	void *work_ipm;

	for(ii=0; ii<N; ii++)
		{
		d_create_strmat(nu[ii]+nx[ii]+1, nx[ii+1], &hsBAbt[ii], (void *) c_ptr);
		c_ptr += hsBAbt[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &hsRSQrq[ii], (void *) c_ptr);
		c_ptr += hsRSQrq[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strmat(nu[ii]+nx[ii], ng[ii], &hsDCt[ii], (void *) c_ptr);
		c_ptr += hsDCt[ii].memory_size;
		}
	
	work_ipm = (void *) c_ptr;
	c_ptr += d_ip2_res_mpc_hard_work_space_size_bytes_libstr(N, nx, nu, nb, ng);

	for(ii=0; ii<N; ii++)
		{
		d_create_strvec(nx[ii+1], &hsb[ii], (void *) c_ptr);
		c_ptr += hsb[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(nu[ii]+nx[ii], &hsrq[ii], (void *) c_ptr);
		c_ptr += hsrq[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(2*nb[ii]+2*ng[ii], &hsd[ii], (void *) c_ptr);
		c_ptr += hsd[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(nu[ii]+nx[ii], &hsux[ii], (void *) c_ptr);
		c_ptr += hsux[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(nx[ii], &hspi[ii], (void *) c_ptr);
		c_ptr += hspi[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(2*nb[ii]+2*ng[ii], &hslam[ii], (void *) c_ptr);
		c_ptr += hslam[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(2*nb[ii]+2*ng[ii], &hst[ii], (void *) c_ptr);
		c_ptr += hst[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(nu[ii]+nx[ii], &hsrrq[ii], (void *) c_ptr);
		c_ptr += hsrrq[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(2*nb[ii]+2*ng[ii], &hsrb[ii], (void *) c_ptr);
		c_ptr += hsrb[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(2*nb[ii]+2*ng[ii], &hsrd[ii], (void *) c_ptr);
		c_ptr += hsrd[ii].memory_size;
		}

	for(ii=0; ii<=N; ii++)
		{
		d_create_strvec(2*nb[ii]+2*ng[ii], &hsrm[ii], (void *) c_ptr);
		c_ptr += hsrm[ii].memory_size;
		}



	// convert matrices

	// TODO use pointers to exploit time invariant !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// dynamic system
	for(ii=0; ii<N; ii++)
		{
		d_cvt_tran_mat2strmat(nx[ii+1], nu[ii], B[ii], nx[ii+1], &hsBAbt[ii], 0, 0);
		d_cvt_tran_mat2strmat(nx[ii+1], nx[ii], A[ii], nx[ii+1], &hsBAbt[ii], nu[ii], 0);
		d_cvt_tran_mat2strmat(nx[ii+1], 1, b[ii], nx[ii+1], &hsBAbt[ii], nu[ii]+nx[ii], 0);
		d_cvt_vec2strvec(nx[ii+1], b[ii], &hsb[ii], 0);
		}
//	for(ii=0; ii<N; ii++)
//		d_print_strmat(nu[ii]+nx[ii]+1, nx[ii+1], &hsBAbt[ii], 0, 0);
//	for(ii=0; ii<N; ii++)
//		d_print_tran_strvec(nx[ii+1], &hsb[ii], 0);
//	exit(1);

	// general constraints
	for(ii=0; ii<N; ii++)
		{
		d_cvt_tran_mat2strmat(ng[ii], nu[ii], D[ii], ng[ii], &hsDCt[ii], 0, 0);
		d_cvt_tran_mat2strmat(ng[ii], nx[ii], C[ii], ng[ii], &hsDCt[ii], nu[ii], 0);
		}
	ii = N;
	d_cvt_tran_mat2strmat(ng[ii], nx[ii], C[ii], ng[ii], &hsDCt[ii], 0, 0);
//	for(ii=0; ii<=N; ii++)
//		d_print_strmat(nu[ii]+nx[ii], ng[ii], &hsDCt[ii], 0, 0);
//	exit(1);

	// cost function
	for(ii=0; ii<N; ii++)
		{
		d_cvt_mat2strmat(nu[ii], nu[ii], R[ii], nu[ii], &hsRSQrq[ii], 0, 0);
		d_cvt_tran_mat2strmat(nu[ii], nx[ii], S[ii], nu[ii], &hsRSQrq[ii], nu[ii], 0);
		d_cvt_mat2strmat(nx[ii], nx[ii], Q[ii], nx[ii], &hsRSQrq[ii], nu[ii], nu[ii]);
		d_cvt_tran_mat2strmat(nu[ii], 1, r[ii], nu[ii], &hsRSQrq[ii], nu[ii]+nx[ii], 0);
		d_cvt_tran_mat2strmat(nx[ii], 1, q[ii], nx[ii], &hsRSQrq[ii], nu[ii]+nx[ii], nu[ii]);
		d_cvt_vec2strvec(nu[ii], r[ii], &hsrq[ii], 0);
		d_cvt_vec2strvec(nx[ii], q[ii], &hsrq[ii], nu[ii]);
		}
	ii = N;
	d_cvt_mat2strmat(nx[ii], nx[ii], Q[ii], nx[ii], &hsRSQrq[ii], 0, 0);
	d_cvt_tran_mat2strmat(nx[ii], 1, q[ii], nx[ii], &hsRSQrq[ii], nx[ii], 0);
	d_cvt_vec2strvec(nx[ii], q[ii], &hsrq[ii], 0);
//	for(ii=0; ii<=N; ii++)
//		d_print_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &hsRSQrq[ii], 0, 0);
//	for(ii=0; ii<=N; ii++)
//		d_print_tran_strvec(nu[ii]+nx[ii], &hsrq[ii], 0);
//	exit(1);

	// estimate mu0 if not user-provided
//	printf("%f\n", mu0);
	if(mu0<=0)
		{
		for(ii=1; ii<N; ii++)
			{
			for(jj=0; jj<nu[ii]; jj++) for(ll=0; ll<nu[ii]; ll++) mu0 = fmax(mu0, R[ii][jj*nu[ii]+ll]);
			for(jj=0; jj<nx[ii]*nu[ii]; jj++) mu0 = fmax(mu0, S[ii][jj]);
			for(jj=0; jj<nx[ii]; jj++) for(ll=0; ll<nx[ii]; ll++) mu0 = fmax(mu0, Q[ii][jj*nx[ii]+ll]);
			for(jj=0; jj<nu[ii]; jj++) mu0 = fmax(mu0, r[ii][jj]);
			for(jj=0; jj<nx[ii]; jj++) mu0 = fmax(mu0, q[ii][jj]);
			}
		ii=N;
		for(jj=0; jj<nx[ii]; jj++) for(ll=0; ll<nx[ii]; ll++) mu0 = fmax(mu0, Q[ii][jj*nx[ii]+ll]);
		for(jj=0; jj<nx[ii]; jj++) mu0 = fmax(mu0, q[ii][jj]);
		}
//	printf("%f\n", mu0);
//	exit(1);

	// TODO how to handle equality constraints?
	// box constraints 
	for(ii=0; ii<=N; ii++)
		{
		d_cvt_vec2strvec(nb[ii], lb[ii], &hsd[ii], 0);
		d_cvt_vec2strvec(nb[ii], ub[ii], &hsd[ii], nb[ii]);
		}
	// general constraints
	for(ii=0; ii<=N; ii++)
		{
		d_cvt_vec2strvec(ng[ii], lg[ii], &hsd[ii], 2*nb[ii]+0);
		d_cvt_vec2strvec(ng[ii], ug[ii], &hsd[ii], 2*nb[ii]+ng[ii]);
		}
//	for(ii=0; ii<=N; ii++)
//		d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hsd[ii], 0);
//	exit(1);

	



#if 0
	if(N2<N) // partial condensing
		{

		// compute partially condensed problem size
		int nx2[N2+1];
		int nu2[N2+1];
		int nb2[N2+1];
		int ng2[N2+1];

		d_part_cond_compute_problem_size(N, nx, nu, nb, hidxb, ng, N2, nx2, nu2, nb2, ng2);



		// data structure of partially condensed system
		double *hpBAbt2[N2];
		double *hpRSQrq2[N2+1];
		double *hpDCt2[N2+1];
		double *hd2[N2+1];
		double *hux2[N2+1];
		double *hpi2[N2];
		double *hlam2[N2+1];
		double *ht2[N2+1];
		void *memory_part_cond;
		void *work_part_cond;
		void *work_part_expand;

		int *hidxb2[N2+1];


		// align (again) to (typical) cache line size
		addr = (( (size_t) ptr ) + 63 ) / 64 * 64;
		ptr = (double *) addr;


		// partial condensing memory and work space
		char_ptr = (char *) ptr;
		memory_part_cond = char_ptr;
		char_ptr += d_part_cond_memory_space_size_bytes(N, nx, nu, nb, hidxb, ng, N2, nx2, nu2, nb2, ng2);

		work_part_cond = char_ptr;
		char_ptr += d_part_cond_work_space_size_bytes(N, nx, nu, nb, hidxb, ng, N2, nx2, nu2, nb2, ng2);
		ptr = (double *) char_ptr;


		// partial condensing routine (computing also hidxb2) !!!
		d_part_cond(N, nx, nu, nb, hidxb, ng, hpBAbt, hpRSQrq, hpDCt, hd, N2, nx2, nu2, nb2, hidxb2, ng2, hpBAbt2, hpRSQrq2, hpDCt2, hd2, memory_part_cond, work_part_cond);


		// packed quantities
		int pnx2[N2+1];
		int pnz2[N2+1];
		int pnb2[N2+1];
		int png2[N2+1];
		int cnx2[N2+1];
		int cnux2[N2+1];
		int cng2[N2+1];

		for(ii=0; ii<=N2; ii++)
			{
			pnx2[ii] = (nx2[ii]+bs-1)/bs*bs;
			pnz2[ii] = (nu2[ii]+nx2[ii]+1+bs-1)/bs*bs;
			pnb2[ii] = (nb2[ii]+bs-1)/bs*bs;
			png2[ii] = (ng2[ii]+bs-1)/bs*bs;
			cnx2[ii] = (nx2[ii]+ncl-1)/ncl*ncl;
			cnux2[ii] = (nu2[ii]+nx2[ii]+ncl-1)/ncl*ncl;
			cng2[ii] = (ng2[ii]+ncl-1)/ncl*ncl;
			}


		// IPM work space
		char_ptr = (char *) ptr;
		work_ipm = char_ptr;
		char_ptr += d_ip2_res_mpc_hard_tv_work_space_size_bytes(N2, nx2, nu2, nb2, ng2);
		ptr = (double *) char_ptr;


		// solution vectors & relative work space
		for(ii=0; ii<=N2; ii++)
			{
			hux2[ii] = ptr;
			ptr += pnz2[ii];
			}

		for(ii=0; ii<N2; ii++)
			{
			hpi2[ii] = ptr;
			ptr += pnx2[ii+1];
			}

		for(ii=0; ii<=N2; ii++)
			{
			hlam2[ii] = ptr;
			ptr += 2*pnb2[ii]+2*png2[ii];
			}

		for(ii=0; ii<=N2; ii++)
			{
			ht2[ii] = ptr;
			ptr += 2*pnb2[ii]+2*png2[ii];
			}



		// initial guess TODO part cond
		if(warm_start)
			{

//			for(ii=0; ii<N; ii++)
//				for(jj=0; jj<nu[ii]; jj++)
//					hux[ii][jj] = u[ii][jj];

//			for(ii=0; ii<=N; ii++)
//				for(jj=0; jj<nx[ii]; jj++)
//					hux[ii][nu[ii]+jj] = x[ii][jj];

			}
//		for(ii=0; ii<=N; ii++)
//			d_print_mat(1, nu[ii]+nx[ii], hux[ii], 1);
//		exit(1);


		// IPM solver on partially condensed system
		hpmpc_status = d_ip2_res_mpc_hard_tv(kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N2, nx2, nu2, nb2, hidxb2, ng2, hpBAbt2, hpRSQrq2, hpDCt2, hd2, hux2, 1, hpi2, hlam2, ht2, work_ipm);

#if 0
		for(ii=0; ii<=N2; ii++)
			d_print_mat(1, nu2[ii]+nx2[ii], hux2[ii], 1);
		exit(2);
#endif


		// expand work space
		char_ptr = (char *) ptr;
		work_part_expand = char_ptr;
		char_ptr += d_part_expand_work_space_size_bytes(N, nx, nu, nb, ng);
		ptr = (double *) char_ptr;


		// expand solution of full space system
		d_part_expand_solution(N, nx, nu, nb, hidxb, ng, hpBAbt, hb, hpRSQrq, hrq, hpDCt, hux, hpi, hlam, ht, N2, nx2, nu2, nb2, hidxb2, ng2, hux2, hpi2, hlam2, ht2, work_part_expand);

#if 0
		for(ii=0; ii<=N; ii++)
			d_print_mat(1, nu[ii]+nx[ii], hux[ii], 1);
		exit(2);
#endif

		}
	else // full space system
		{
#endif


		// initial guess
		if(warm_start)
			{

			for(ii=0; ii<N; ii++)
				d_cvt_vec2strvec(nu[ii], u[ii], &hsux[ii], 0);

			for(ii=0; ii<=N; ii++)
				d_cvt_vec2strvec(nx[ii], x[ii], &hsux[ii], nu[ii]);

			}

#if 0
		printf("\n%d\n", N);
		for(ii=0; ii<=N; ii++)
			printf("\n%d %d\n", nu[ii], nx[ii]);
//			d_print_mat(1, nu[ii]+nx[ii], hux[ii], 1);
		exit(1);
#endif



		// IPM solver on full space system
		hpmpc_status = d_ip2_res_mpc_hard_libstr(kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx, nu, nb, hidxb, ng, hsBAbt, hsRSQrq, hsDCt, hsd, hsux, 1, hspi, hslam, hst, work_ipm);
//		hpmpc_status = d_ip2_res_mpc_hard_tv(kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx, nu, nb, hidxb, ng, hpBAbt, hpRSQrq, hpDCt, hd, hux, 1, hpi, hlam, ht, work_ipm);

#if 0
		}
#endif

//	for(ii=0; ii<=N; ii++)
//		d_print_tran_strvec(nu[ii]+nx[ii], &hsux[ii], 0);
//	exit(1);



	// copy back inputs and states
	for(ii=0; ii<N; ii++)
		d_cvt_strvec2vec(nu[ii], &hsux[ii], 0, u[ii]);

	for(ii=0; ii<=N; ii++)
		d_cvt_strvec2vec(nx[ii], &hsux[ii], nu[ii], x[ii]);



#if 0
	// compute infinity norm of residuals on exit

	double mu;

	d_res_mpc_hard_tv(N, nx, nu, nb, hidxb, ng, hpBAbt, hb, hpRSQrq, hrq, hux, hpDCt, hd, hpi, hlam, ht, hrrq, hrb, hrd, &mu);

	temp = fabs(hrrq[0][0]);
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nu[ii]+nx[ii]; jj++) 
			temp = fmax( temp, fabs(hrrq[ii][jj]) );
	ii = N;
	for(jj=0; jj<nx[ii]; jj++) 
		temp = fmax( temp, fabs(hrrq[ii][jj]) );
	inf_norm_res[0] = temp;

	temp = fabs(hrb[0][0]);
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx[ii+1]; jj++) 
			temp = fmax( temp, fabs(hrb[ii][jj]) );
	inf_norm_res[1] = temp;

	temp = fabs(hrd[0][0]);
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<nb[ii]; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][jj+0]) );
			temp = fmax( temp, fabs(hrd[ii][jj+pnb[ii]]) );
			}
		}
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<ng[ii]; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][2*pnb[ii]+jj+0]) );
			temp = fmax( temp, fabs(hrd[ii][2*pnb[ii]+jj+png[ii]]) );
			}
		}
	inf_norm_res[2] = temp;

	inf_norm_res[3] = mu;



	// copy back multipliers

	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx[ii+1]; jj++)
			pi[ii][jj] = hpi[ii][jj];

	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<nb[ii]; jj++)
			{
			lam[ii][jj+0]      = hlam[ii][jj+0];
			lam[ii][jj+nb[ii]] = hlam[ii][jj+pnb[ii]];
//			t[ii][jj+0]      = ht[ii][jj+0];
//			t[ii][jj+nb[ii]] = ht[ii][jj+pnb[ii]];
			}
		}

	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<ng[ii]; jj++)
			{
			lam[ii][2*nb[ii]+jj+0]      = hlam[ii][2*pnb[ii]+jj+0];
			lam[ii][2*nb[ii]+jj+ng[ii]] = hlam[ii][2*pnb[ii]+jj+png[ii]];
//			t[ii][2*nb[ii]+jj+0]      = ht[ii][2*pnb[ii]+jj+0];
//			t[ii][2*nb[ii]+jj+ng[ii]] = ht[ii][2*pnb[ii]+jj+png[ii]];
			}
		}

//	printf("\nend of wrapper\n");
#endif

    return hpmpc_status;

	}



