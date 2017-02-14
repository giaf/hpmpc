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


#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX

#ifdef BLASFEO

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_blas.h>
#include <blasfeo_d_aux.h>



int d_res_res_mpc_hard_work_space_size_bytes_libstr(int N, int *nx, int *nu, int *nb, int *ng)
	{

	int ii;

	int ngM = 0;
	for(ii=0; ii<=N; ii++)
		{
		ngM = ng[ii]>ngM ? ng[ii] : ngM;
		}

	int size = 0;

	size += 2*d_size_strvec(ngM); // res_work[0], res_work[1]

	// make multiple of (typical) cache line size
	size = (size+63)/64*64;

	return size;

	}



void d_res_res_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strmat *hsQ, struct d_strvec *hsq, struct d_strvec *hsux, struct d_strmat *hsDCt, struct d_strvec *hsd, struct d_strvec *hspi, struct d_strvec *hslam, struct d_strvec *hst, struct d_strvec *hsrq, struct d_strvec *hsrb, struct d_strvec *hsrd, struct d_strvec *hsrm, double *mu, void *work)
	{

	int ii, jj;

	char *c_ptr;

	struct d_strvec hswork_0, hswork_1;
	double *work0, *work1;

	double
		*ptr_b, *ptr_q, *ptr_d, *ptr_ux, *ptr_pi, *ptr_lam, *ptr_t, *ptr_rb, *ptr_rq, *ptr_rd, *ptr_rm;
	
	int
		*ptr_idxb;
	
	int nu0, nu1, nx0, nx1, nxm, nb0, ng0, nb_tot;

	__m256d
		v_ux, v_tmp2,
		v_tmp0, v_lam0, v_t0, v_mu0,
		v_tmp1, v_lam1, v_t1, v_mu1;
	
	__m128d
		u_ux, u_tmp2,
		u_tmp0, u_lam0, u_t0, u_mu0,
		u_tmp1, u_lam1, u_t1, u_mu1;

	double
		mu2;

	// initialize mu
	nb_tot = 0;
	mu2 = 0;
	u_mu0 = _mm_setzero_pd();
	u_mu1 = _mm_setzero_pd();
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();

	// loop over stages
	for(ii=0; ii<=N; ii++)
		{
		nu0 = nu[ii];
		nx0 = nx[ii];
		nb0 = nb[ii];
		ng0 = ng[ii];

		ptr_b = hsb[ii].pa;
		ptr_q = hsq[ii].pa;
		ptr_ux = hsux[ii].pa;
		ptr_pi = hspi[ii].pa;
		ptr_rq = hsrq[ii].pa;
		ptr_rb = hsrb[ii].pa;

		if(nb0>0 | ng0>0)
			{
			ptr_d = hsd[ii].pa;
			ptr_lam = hslam[ii].pa;
			ptr_t = hst[ii].pa;
			ptr_rd = hsrd[ii].pa;
			ptr_rm = hsrm[ii].pa;
			}

		dveccp_libstr(nu0+nx0, 1.0, &hsq[ii], 0, &hsrq[ii], 0);

		// no previous multiplier at the first stage
		if(ii>0)
			daxpy_libstr(nx0, -1.0, &hspi[ii], 0, &hsrq[ii], nu[ii], &hsrq[ii], nu[ii]);

		if(nb0>0)
			{

			ptr_idxb = idxb[ii];
			nb_tot += nb0;

			for(jj=0; jj<nb0-3; jj+=4)
				{
				v_lam0 = _mm256_loadu_pd( &ptr_lam[jj+0] );
				v_lam1 = _mm256_loadu_pd( &ptr_lam[jj+nb0] );
				v_tmp0 = _mm256_sub_pd( v_lam1, v_lam0 );

				u_tmp0 = _mm_load_sd( &ptr_rq[ptr_idxb[jj+0]] );
				u_tmp1 = _mm_load_sd( &ptr_rq[ptr_idxb[jj+2]] );
				u_tmp0 = _mm_loadh_pd( u_tmp0, &ptr_rq[ptr_idxb[jj+1]] );
				u_tmp1 = _mm_loadh_pd( u_tmp1, &ptr_rq[ptr_idxb[jj+3]] );
				v_tmp1 = _mm256_castpd128_pd256( u_tmp0 );
				v_tmp1 = _mm256_insertf128_pd( v_tmp1, u_tmp1, 0x1 );

				v_tmp0 = _mm256_add_pd( v_tmp0, v_tmp1 );

				u_tmp1 = _mm256_extractf128_pd( v_tmp0, 0x1 );
				u_tmp0 = _mm256_castpd256_pd128( v_tmp0 );

				_mm_store_sd( &ptr_rq[ptr_idxb[jj+0]], u_tmp0 );
				_mm_storeh_pd( &ptr_rq[ptr_idxb[jj+1]], u_tmp0 );
				_mm_store_sd( &ptr_rq[ptr_idxb[jj+2]], u_tmp1 );
				_mm_storeh_pd( &ptr_rq[ptr_idxb[jj+3]], u_tmp1 );

				u_tmp0 = _mm_load_sd( &ptr_ux[ptr_idxb[jj+0]] );
				u_tmp1 = _mm_load_sd( &ptr_ux[ptr_idxb[jj+2]] );
				u_tmp0 = _mm_loadh_pd( u_tmp0, &ptr_ux[ptr_idxb[jj+1]] );
				u_tmp1 = _mm_loadh_pd( u_tmp1, &ptr_ux[ptr_idxb[jj+3]] );
				v_ux   = _mm256_castpd128_pd256( u_tmp0 );
				v_ux   = _mm256_insertf128_pd( v_ux, u_tmp1, 0x1 );

				v_t0   = _mm256_loadu_pd( &ptr_t[jj+0] );
				v_t1   = _mm256_loadu_pd( &ptr_t[jj+nb0] );

				v_tmp0   = _mm256_loadu_pd( &ptr_d[jj+0] );
				v_tmp1   = _mm256_loadu_pd( &ptr_d[jj+nb0] );
				v_tmp0   = _mm256_sub_pd( v_tmp0, v_ux );
				v_tmp1   = _mm256_sub_pd( v_tmp1, v_ux );
				v_tmp0   = _mm256_add_pd( v_tmp0, v_t0 );
				v_tmp1   = _mm256_sub_pd( v_tmp1, v_t1 );
				_mm256_storeu_pd( &ptr_rd[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rd[jj+nb0], v_tmp1 );

				v_tmp0 = _mm256_mul_pd( v_lam0, v_t0 );
				v_tmp1 = _mm256_mul_pd( v_lam1, v_t1 );
				_mm256_storeu_pd( &ptr_rm[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rm[jj+nb0], v_tmp1 );
				v_mu0  = _mm256_add_pd( v_mu0, v_tmp0 );
				v_mu1  = _mm256_add_pd( v_mu1, v_tmp1 );
				}
			for(; jj<nb0; jj++)
				{
				u_lam0 = _mm_load_sd( &ptr_lam[jj+0] );
				u_lam1 = _mm_load_sd( &ptr_lam[jj+nb0] );
				u_tmp0 = _mm_sub_sd( u_lam1, u_lam0 );
				u_tmp1 = _mm_load_sd( &ptr_rq[ptr_idxb[jj]] );
				u_tmp0 = _mm_add_sd( u_tmp0, u_tmp1 );
				_mm_store_sd( &ptr_rq[ptr_idxb[jj]], u_tmp0 );

				u_ux   = _mm_load_sd( &ptr_ux[ptr_idxb[jj]] );

				u_t0   = _mm_load_sd( &ptr_t[jj+0] );
				u_t1   = _mm_load_sd( &ptr_t[jj+nb0] );

				u_tmp0   = _mm_load_sd( &ptr_d[jj+0] );
				u_tmp1   = _mm_load_sd( &ptr_d[jj+nb0] );
				u_tmp0   = _mm_sub_sd( u_tmp0, u_ux );
				u_tmp1   = _mm_sub_sd( u_tmp1, u_ux );
				u_tmp0   = _mm_add_sd( u_tmp0, u_t0 );
				u_tmp1   = _mm_sub_sd( u_tmp1, u_t1 );
				_mm_store_sd( &ptr_rd[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rd[jj+nb0], u_tmp1 );

				u_tmp0 = _mm_mul_sd( u_lam0, u_t0 );
				u_tmp1 = _mm_mul_sd( u_lam1, u_t1 );
				_mm_store_sd( &ptr_rm[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rm[jj+nb0], u_tmp1 );
				u_mu0  = _mm_add_sd( u_mu0, u_tmp0 );
				u_mu1  = _mm_add_sd( u_mu1, u_tmp1 );
				}
			}

		dsymv_l_libstr(nu0+nx0, nu0+nx0, 1.0, &hsQ[ii], 0, 0, &hsux[ii], 0, 1.0, &hsrq[ii], 0, &hsrq[ii], 0);

		// no dynamic at the last stage
		if(ii<N)
			{
			nu1 = nu[ii+1];
			nx1 = nx[ii+1];

			daxpy_libstr(nx1, -1.0, &hsux[ii+1], nu1, &hsb[ii], 0, &hsrb[ii], 0);

			dgemv_nt_libstr(nu0+nx0, nx1, 1.0, 1.0, &hsBAbt[ii], 0, 0, &hspi[ii+1], 0, &hsux[ii], 0, 1.0, 1.0, &hsrq[ii], 0, &hsrb[ii], 0, &hsrq[ii], 0, &hsrb[ii], 0);
			}

		if(ng0>0)
			{

			c_ptr = (char *) work;
			d_create_strvec(ng0, &hswork_0, (void *) c_ptr);
			c_ptr += hswork_0.memory_size;
			d_create_strvec(ng0, &hswork_1, (void *) c_ptr);
			c_ptr += hswork_1.memory_size;
			work0 = hswork_0.pa;
			work1 = hswork_1.pa;

			ptr_d   += 2*nb0;
			ptr_lam += 2*nb0;
			ptr_t   += 2*nb0;
			ptr_rd  += 2*nb0;
			ptr_rm  += 2*nb0;

			nb_tot += ng0;

			for(jj=0; jj<ng0-3; jj+=4)
				{
				v_lam0 = _mm256_loadu_pd( &ptr_lam[jj+0] );
				v_lam1 = _mm256_loadu_pd( &ptr_lam[jj+ng0] );
				v_tmp0 = _mm256_sub_pd( v_lam1, v_lam0 );
				_mm256_storeu_pd( &work0[jj], v_tmp0 );

				v_t0   = _mm256_loadu_pd( &ptr_t[jj+0] );
				v_t1   = _mm256_loadu_pd( &ptr_t[jj+ng0] );
				v_tmp0 = _mm256_loadu_pd( &ptr_d[jj+0] );
				v_tmp1 = _mm256_loadu_pd( &ptr_d[jj+ng0] );
				v_tmp0 = _mm256_add_pd( v_tmp0, v_t0 );
				v_tmp1 = _mm256_sub_pd( v_tmp1, v_t1 );
				_mm256_storeu_pd( &ptr_rd[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rd[jj+ng0], v_tmp1 );

				v_tmp0 = _mm256_mul_pd( v_lam0, v_t0 );
				v_tmp1 = _mm256_mul_pd( v_lam1, v_t1 );
				_mm256_storeu_pd( &ptr_rm[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rm[jj+ng0], v_tmp1 );
				v_mu0  = _mm256_add_pd( v_mu0, v_tmp0 );
				v_mu1  = _mm256_add_pd( v_mu1, v_tmp1 );
				}
			for(; jj<ng0; jj++) // TODO mask ?
				{
				u_lam0 = _mm_load_sd( &ptr_lam[jj+0] );
				u_lam1 = _mm_load_sd( &ptr_lam[jj+ng0] );
				u_tmp0 = _mm_sub_sd( u_lam1, u_lam0 );
				_mm_store_sd( &work0[jj], u_tmp0 );

				u_t0   = _mm_load_sd( &ptr_t[jj+0] );
				u_t1   = _mm_load_sd( &ptr_t[jj+ng0] );
				u_tmp0 = _mm_load_sd( &ptr_d[jj+0] );
				u_tmp1 = _mm_load_sd( &ptr_d[jj+ng0] );
				u_tmp0 = _mm_add_sd( u_tmp0, u_t0 );
				u_tmp1 = _mm_sub_sd( u_tmp1, u_t1 );
				_mm_store_sd( &ptr_rd[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rd[jj+ng0], u_tmp1 );

				u_tmp0 = _mm_mul_sd( u_lam0, u_t0 );
				u_tmp1 = _mm_mul_sd( u_lam1, u_t1 );
				_mm_store_sd( &ptr_rm[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rm[jj+ng0], u_tmp1 );
				u_mu0  = _mm_add_sd( u_mu0, u_tmp0 );
				u_mu1  = _mm_add_sd( u_mu1, u_tmp1 );
				}

			dgemv_nt_libstr(nu0+nx0, ng0, 1.0, 1.0, &hsDCt[ii], 0, 0, &hswork_0, 0, &hsux[ii], 0, 1.0, 0.0, &hsrq[ii], 0, &hswork_1, 0, &hsrq[ii], 0, &hswork_1, 0);

			for(jj=0; jj<ng0-3; jj+=4)
				{
				v_tmp2 = _mm256_loadu_pd( &work1[jj] );
				v_tmp0 = _mm256_loadu_pd( &ptr_rd[jj+0] );
				v_tmp1 = _mm256_loadu_pd( &ptr_rd[jj+ng0] );
				v_tmp0 = _mm256_sub_pd( v_tmp0, v_tmp2 );
				v_tmp1 = _mm256_sub_pd( v_tmp1, v_tmp2 );
				_mm256_storeu_pd( &ptr_rd[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rd[jj+ng0], v_tmp1 );
				}
			for(; jj<ng0; jj++) // TODO mask ?
				{
				u_tmp2 = _mm_load_sd( &work1[jj] );
				u_tmp0 = _mm_load_sd( &ptr_rd[jj+0] );
				u_tmp1 = _mm_load_sd( &ptr_rd[jj+ng0] );
				u_tmp0 = _mm_sub_sd( u_tmp0, u_tmp2 );
				u_tmp1 = _mm_sub_sd( u_tmp1, u_tmp2 );
				_mm_store_sd( &ptr_rd[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rd[jj+ng0], u_tmp1 );
				}

			}

		}
	
	// normalize mu
	double mu_scal = 0.0;
	if(nb_tot!=0)
		{
		mu_scal = 1.0 / (2.0*nb_tot);

		v_mu0  = _mm256_add_pd( v_mu0, v_mu1 );
		u_mu0  = _mm_add_sd( u_mu0, u_mu1 );
		u_tmp0 = _mm_add_pd( _mm256_castpd256_pd128( v_mu0 ), _mm256_extractf128_pd( v_mu0, 0x1 ) );
		u_tmp0 = _mm_hadd_pd( u_tmp0, u_tmp0);
		u_mu0  = _mm_add_sd( u_mu0, u_tmp0 );
		u_mu1  = _mm_load_sd( &mu2 );
		u_mu0  = _mm_add_sd( u_mu0, u_mu1 );
		u_tmp0 = _mm_load_sd( &mu_scal );
		u_mu0  = _mm_mul_sd( u_mu0, u_tmp0 );
		_mm_store_sd( &mu[0], u_mu0 );
		}

	return;

	}



#endif
