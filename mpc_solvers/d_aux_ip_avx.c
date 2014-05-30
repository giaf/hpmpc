/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     *
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



void d_update_hessian_box(int k0, int kmax, int nb, int cnz, double sigma_mu, double *t, double *lam, double *lamt, double *dlam, double *bd, double *bl, double *pd, double *pl, double *db)
	{
	
	__m256d
		v_ones, v_sigma_mu,
		v_tmp, v_lam, v_lamt, v_dlam, v_db;
		
	__m128d
		u_lamt, u_bd, u_bl, u_lam;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );
	
	const int bs = 4; //d_get_mr();
	
	double temp0, temp1;
	
	int ii, ll, bs0;
	
	t    += k0;
	lam  += k0;
	lamt += k0;
	dlam += k0;
	
	ii=k0; // k0 supposed to be multiple of 2*bs !!!!!!!!!!
	for(; ii<kmax-3; ii+=4)
		{
		
		v_tmp  = _mm256_load_pd( &t[0] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		v_lam  = _mm256_load_pd( &lam[0] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &lamt[0], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &dlam[0], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[ii] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[0+(ii+0)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[1+(ii+1)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[2*ii+0] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[ii] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[(ii+0)*bs], u_bl );
		_mm_storeh_pd( &pl[(ii+1)*bs], u_bl );


		v_tmp  = _mm256_load_pd( &t[4] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		v_lam  = _mm256_load_pd( &lam[4] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &lamt[4], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &dlam[4], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[ii+2] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[2+(ii+2)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[3+(ii+3)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[2*ii+4] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[ii+2] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[(ii+2)*bs], u_bl );
		_mm_storeh_pd( &pl[(ii+3)*bs], u_bl );


/*		temp0 = 1.0/t[0];*/
/*		temp1 = 1.0/t[1];*/
/*		lamt[0] = lam[0]*temp0;*/
/*		lamt[1] = lam[1]*temp1;*/
/*		dlam[0] = temp0*sigma_mu; // !!!!!*/
/*		dlam[1] = temp1*sigma_mu; // !!!!!*/
/*		pd[0+(ii+0)*bs+ii*cnz] = bd[ii+0] + lamt[0] + lamt[1];*/
/*		pl[(ii+0)*bs] = bl[ii+0] + lam[1] + lamt[1]*db[2*ii+1] + dlam[1] */
/*		                         - lam[0] - lamt[0]*db[2*ii+0] - dlam[0];*/

/*		temp0 = 1.0/t[2];*/
/*		temp1 = 1.0/t[3];*/
/*		lamt[2] = lam[2]*temp0;*/
/*		lamt[3] = lam[3]*temp1;*/
/*		dlam[2] = temp0*sigma_mu; // !!!!!*/
/*		dlam[3] = temp1*sigma_mu; // !!!!!*/
/*		pd[1+(ii+1)*bs+ii*cnz] = bd[ii+1] + lamt[2] + lamt[3];*/
/*		pl[(ii+1)*bs] = bl[ii+1] + lam[3] + lamt[3]*db[2*ii+3] + dlam[3] */
/*		                         - lam[2] - lamt[2]*db[2*ii+2] - dlam[2];*/

/*		temp0 = 1.0/t[4];*/
/*		temp1 = 1.0/t[5];*/
/*		lamt[4] = lam[4]*temp0;*/
/*		lamt[5] = lam[5]*temp1;*/
/*		dlam[4] = temp0*sigma_mu; // !!!!!*/
/*		dlam[5] = temp1*sigma_mu; // !!!!!*/
/*		pd[2+(ii+2)*bs+ii*cnz] = bd[ii+2] + lamt[4] + lamt[5];*/
/*		pl[(ii+2)*bs] = bl[ii+2] + lam[5] + lamt[5]*db[2*ii+5] + dlam[5] */
/*		                         - lam[4] - lamt[4]*db[2*ii+4] - dlam[4];*/

/*		temp0 = 1.0/t[6];*/
/*		temp1 = 1.0/t[7];*/
/*		lamt[6] = lam[6]*temp0;*/
/*		lamt[7] = lam[7]*temp1;*/
/*		dlam[6] = temp0*sigma_mu; // !!!!!*/
/*		dlam[7] = temp1*sigma_mu; // !!!!!*/
/*		pd[3+(ii+3)*bs+ii*cnz] = bd[ii+3] + lamt[6] + lamt[7];*/
/*		pl[(ii+3)*bs] = bl[ii+3] + lam[7] + lamt[7]*db[2*ii+7] + dlam[7] */
/*		                         - lam[6] - lamt[6]*db[2*ii+6] - dlam[6];*/

		t    += 8;
		lam  += 8;
		lamt += 8;
		dlam += 8;

		}
	if(ii<kmax)
		{
		bs0 = nb-ii;
		for(ll=0; ll<bs0; ll++)
			{
			temp0 = 1.0/t[0];
			temp1 = 1.0/t[1];
			lamt[0] = lam[0]*temp0;
			lamt[1] = lam[1]*temp1;
			dlam[0] = temp0*sigma_mu; // !!!!!
			dlam[1] = temp1*sigma_mu; // !!!!!
			pd[ll+(ii+ll)*bs+ii*cnz] = bd[ii+ll] + lamt[0] + lamt[1];
			pl[(ii+ll)*bs] = bl[ii+ll] + lam[1] + lamt[1]*db[2*ii+2*ll+1] + dlam[1] 
			                           - lam[0] - lamt[0]*db[2*ii+2*ll+0] - dlam[0];

			t    += 2;
			lam  += 2;
			lamt += 2;
			dlam += 2;
			}
		}

	}



/*void d_update_hessian_box(int k0, int kmax, int nb, int cnz, double sigma_mu, double *t, double *lam, double *lamt, double *dlam, double *bd, double *bl, double *pd, double *pl, double *lb, double *ub)*/
/*	{*/
/*	*/
/*	const int bs = 4; //d_get_mr();*/
/*	*/
/*	double temp0, temp1;*/
/*	*/
/*	int ii, ll, bs0;*/
/*	*/
/*	t    += k0;*/
/*	lam  += k0;*/
/*	lamt += k0;*/
/*	dlam += k0;*/
/*	*/
/*	ii=k0; // k0 supposed to be multiple of 2*bs !!!!!!!!!!*/
/*	for(; ii<kmax-3; ii+=4)*/
/*		{*/

/*		temp0 = 1.0/t[0];*/
/*		temp1 = 1.0/t[1];*/
/*		lamt[0] = lam[0]*temp0;*/
/*		lamt[1] = lam[1]*temp1;*/
/*		dlam[0] = temp0*sigma_mu; // !!!!!*/
/*		dlam[1] = temp1*sigma_mu; // !!!!!*/
/*		pd[0+(ii+0)*bs+ii*cnz] = bd[ii+0] + lamt[0] + lamt[1];*/
/*		pl[(ii+0)*bs] = bl[ii+0] + lam[1] - lamt[1]*ub[ii+0] + dlam[1] - lam[0] - lamt[0]*lb[ii+0] - dlam[0];*/

/*		temp0 = 1.0/t[2];*/
/*		temp1 = 1.0/t[3];*/
/*		lamt[2] = lam[2]*temp0;*/
/*		lamt[3] = lam[3]*temp1;*/
/*		dlam[2] = temp0*sigma_mu; // !!!!!*/
/*		dlam[3] = temp1*sigma_mu; // !!!!!*/
/*		pd[1+(ii+1)*bs+ii*cnz] = bd[ii+1] + lamt[2] + lamt[3];*/
/*		pl[(ii+1)*bs] = bl[ii+1] + lam[3] - lamt[3]*ub[ii+1] + dlam[3] - lam[2] - lamt[2]*lb[ii+1] - dlam[2];*/

/*		temp0 = 1.0/t[4];*/
/*		temp1 = 1.0/t[5];*/
/*		lamt[4] = lam[4]*temp0;*/
/*		lamt[5] = lam[5]*temp1;*/
/*		dlam[4] = temp0*sigma_mu; // !!!!!*/
/*		dlam[5] = temp1*sigma_mu; // !!!!!*/
/*		pd[2+(ii+2)*bs+ii*cnz] = bd[ii+2] + lamt[4] + lamt[5];*/
/*		pl[(ii+2)*bs] = bl[ii+2] + lam[5] - lamt[5]*ub[ii+2] + dlam[5] - lam[4] - lamt[4]*lb[ii+2] - dlam[4];*/

/*		temp0 = 1.0/t[6];*/
/*		temp1 = 1.0/t[7];*/
/*		lamt[6] = lam[6]*temp0;*/
/*		lamt[7] = lam[7]*temp1;*/
/*		dlam[6] = temp0*sigma_mu; // !!!!!*/
/*		dlam[7] = temp1*sigma_mu; // !!!!!*/
/*		pd[3+(ii+3)*bs+ii*cnz] = bd[ii+3] + lamt[6] + lamt[7];*/
/*		pl[(ii+3)*bs] = bl[ii+3] + lam[7] - lamt[7]*ub[ii+3] + dlam[7] - lam[6] - lamt[6]*lb[ii+3] - dlam[6];*/

/*		t    += 8;*/
/*		lam  += 8;*/
/*		lamt += 8;*/
/*		dlam += 8;*/

/*		}*/
/*	if(ii<kmax)*/
/*		{*/
/*		bs0 = nb-ii;*/
/*		for(ll=0; ll<bs0; ll++)*/
/*			{*/
/*			temp0 = 1.0/t[0];*/
/*			temp1 = 1.0/t[1];*/
/*			lamt[0] = lam[0]*temp0;*/
/*			lamt[1] = lam[1]*temp1;*/
/*			dlam[0] = temp0*sigma_mu; // !!!!!*/
/*			dlam[1] = temp1*sigma_mu; // !!!!!*/
/*			pd[ll+(ii+ll)*bs+ii*cnz] = bd[ii+ll] + lamt[0] + lamt[1];*/
/*			pl[(ii+ll)*bs] = bl[ii+ll] + lam[1] - lamt[1]*ub[ii+ll] + dlam[1] - lam[0] - lamt[0]*lb[ii+ll] - dlam[0];*/

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/
/*			}*/
/*		}*/

/*	}*/
