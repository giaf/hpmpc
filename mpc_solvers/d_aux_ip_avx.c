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



void d_update_hessian_box_mpc(int N, int k0, int k1, int kmax, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **pl2, double **db)
	{

	__m256d
		v_ones, v_sigma_mu,
		v_tmp, v_lam, v_lamt, v_dlam, v_db;
		
	__m128d
		u_tmp, u_lamt, u_bd, u_bl, u_lam, u_dlam, u_db;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );
	
	const int bs = 4; //d_get_mr();
	
	double temp0, temp1;
	
	double *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_t_inv;
	
	int ii, jj, ll, bs0;
	
	// first stage
	
	ptr_t     = t[0];
	ptr_lam   = lam[0];
	ptr_lamt  = lamt[0];
	ptr_dlam  = dlam[0];
	ptr_t_inv = t_inv[0];
	
	ii = 0;
	for(; ii<k0-3; ii+=4)
		{
		
		v_tmp  = _mm256_load_pd( &ptr_t[0] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[0] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[0], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[0][ii] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[0][0+(ii+0)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[0][1+(ii+1)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[0][2*ii+0] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[0][ii] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[0][(ii+0)*bs], u_bl );
		_mm_storeh_pd( &pl[0][(ii+1)*bs], u_bl );
		_mm_store_pd( &pl2[0][ii+0], u_bl );

		v_tmp  = _mm256_load_pd( &ptr_t[4] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[4], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[4] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[4], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[4], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[0][ii+2] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[0][2+(ii+2)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[0][3+(ii+3)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[0][2*ii+4] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[0][ii+2] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[0][(ii+2)*bs], u_bl );
		_mm_storeh_pd( &pl[0][(ii+3)*bs], u_bl );
		_mm_store_pd( &pl2[jj][ii+2], u_bl );


		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;

		}
	if(ii<k0)
		{

/*		bs0 = nb-ii;*/
		bs0 = k0-ii;
		ll = 0;
		
		if(bs0>=2)
			{

			v_tmp  = _mm256_load_pd( &ptr_t[0] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[0], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[0][ii] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[0][0+(ii+0)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[0][1+(ii+1)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[0][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[0][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[0][(ii+0)*bs], u_bl );
			_mm_storeh_pd( &pl[0][(ii+1)*bs], u_bl );
			_mm_store_pd( &pl2[0][ii+0], u_bl );

			ptr_t     += 4;
			ptr_lam   += 4;
			ptr_lamt  += 4;
			ptr_dlam  += 4;
			ptr_t_inv += 4;
			
			ll   += 2;
			bs0  -= 2;

			}
		
		if(bs0>0)
			{
			
			u_tmp  = _mm_load_pd( &ptr_t[0] );
			u_tmp  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp );
			_mm_store_pd( &ptr_t_inv[0], u_tmp ); // store t_inv
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );
			u_tmp  = _mm_hadd_pd( u_lamt, u_lamt ); // [ lamt[0]+lamt[1] , xxx ]
			u_bd   = _mm_load_sd( &bd[0][ii+ll] );
			u_bd   = _mm_add_sd( u_bd, u_tmp );
			_mm_store_sd( &pd[0][ll+(ii+ll)*bs+ii*cnz], u_bd );
			u_db   = _mm_load_pd( &db[0][2*ii+2*ll+0] );
			u_db   = _mm_mul_pd( u_db, u_lamt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_add_pd( u_lam, u_db );
			u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
			u_bl   = _mm_load_sd( &bl[0][ii+ll] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_sd( &pl[0][(ii+ll)*bs], u_bl );
			_mm_store_sd( &pl2[0][ii+ll], u_bl );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/

			}
		}
	
	// middle stages
	for(jj=1; jj<N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_t_inv = t_inv[jj];

		ii = 0;
		for(; ii<kmax-3; ii+=4)
			{
		
			v_tmp  = _mm256_load_pd( &ptr_t[0] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[0], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[jj][ii] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[jj][0+(ii+0)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[jj][1+(ii+1)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[jj][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[jj][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[jj][(ii+0)*bs], u_bl );
			_mm_storeh_pd( &pl[jj][(ii+1)*bs], u_bl );
			_mm_store_pd( &pl2[jj][ii+0], u_bl );

			v_tmp  = _mm256_load_pd( &ptr_t[4] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[4], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[4] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[4], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[4], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[jj][ii+2] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[jj][2+(ii+2)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[jj][3+(ii+3)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[jj][2*ii+4] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[jj][ii+2] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[jj][(ii+2)*bs], u_bl );
			_mm_storeh_pd( &pl[jj][(ii+3)*bs], u_bl );
			_mm_store_pd( &pl2[jj][ii+2], u_bl );


			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_t_inv += 8;

			}
		if(ii<kmax)
			{

/*			bs0 = nb-ii;*/
			bs0 = kmax-ii;
			ll = 0;
		
			if(bs0>=2)
				{

				v_tmp  = _mm256_load_pd( &ptr_t[0] );
				v_tmp  = _mm256_div_pd( v_ones, v_tmp );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
				v_lam  = _mm256_load_pd( &ptr_lam[0] );
				v_lamt = _mm256_mul_pd( v_tmp, v_lam );
				_mm256_store_pd( &ptr_lamt[0], v_lamt );
				v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam );
				u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
				u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
				u_bd   = _mm_load_pd( &bd[jj][ii] );
				u_bd   = _mm_add_pd( u_bd, u_lamt );
				_mm_storel_pd( &pd[jj][0+(ii+0)*bs+ii*cnz], u_bd );
				_mm_storeh_pd( &pd[jj][1+(ii+1)*bs+ii*cnz], u_bd );
				v_db   = _mm256_load_pd( &db[jj][2*ii+0] );
				v_db   = _mm256_mul_pd( v_db, v_lamt );
				v_lam  = _mm256_add_pd( v_lam, v_dlam );
				v_lam  = _mm256_add_pd( v_lam, v_db );
				u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
				u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
				u_bl   = _mm_load_pd( &bl[jj][ii] );
				u_bl   = _mm_sub_pd( u_bl, u_lam );
				_mm_storel_pd( &pl[jj][(ii+0)*bs], u_bl );
				_mm_storeh_pd( &pl[jj][(ii+1)*bs], u_bl );
				_mm_store_pd( &pl2[jj][ii+0], u_bl );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;
			
				ll   += 2;
				bs0  -= 2;

				}
		
			if(bs0>0)
				{
			
				u_tmp  = _mm_load_pd( &ptr_t[0] );
				u_tmp  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp );
				_mm_store_pd( &ptr_t_inv[0], u_tmp ); // store t_inv
				u_lam  = _mm_load_pd( &ptr_lam[0] );
				u_lamt = _mm_mul_pd( u_tmp, u_lam );
				_mm_store_pd( &ptr_lamt[0], u_lamt );
				u_dlam = _mm_mul_pd( u_tmp, _mm256_castpd256_pd128( v_sigma_mu ) );
				_mm_store_pd( &ptr_dlam[0], u_dlam );
				u_tmp  = _mm_hadd_pd( u_lamt, u_lamt ); // [ lamt[0]+lamt[1] , xxx ]
				u_bd   = _mm_load_sd( &bd[jj][ii+ll] );
				u_bd   = _mm_add_sd( u_bd, u_tmp );
				_mm_store_sd( &pd[jj][ll+(ii+ll)*bs+ii*cnz], u_bd );
				u_db   = _mm_load_pd( &db[jj][2*ii+2*ll+0] );
				u_db   = _mm_mul_pd( u_db, u_lamt );
				u_lam  = _mm_add_pd( u_lam, u_dlam );
				u_lam  = _mm_add_pd( u_lam, u_db );
				u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
				u_bl   = _mm_load_sd( &bl[jj][ii+ll] );
				u_bl   = _mm_sub_pd( u_bl, u_lam );
				_mm_store_sd( &pl[jj][(ii+ll)*bs], u_bl );
				_mm_store_pd( &pl2[jj][ii+ll], u_bl );

	/*			t    += 2;*/
	/*			lam  += 2;*/
	/*			lamt += 2;*/
	/*			dlam += 2;*/

				}
			}
		
		}


	ptr_t     = t[N]     + 2*k1;
	ptr_lam   = lam[N]   + 2*k1;
	ptr_lamt  = lamt[N]  + 2*k1;
	ptr_dlam  = dlam[N]  + 2*k1;
	ptr_t_inv = t_inv[N] + 2*k1;

	ii=k1; // k0 supposed to be multiple of bs !!!!!!!!!!

	// last stage
	for(; ii<kmax-3; ii+=4)
		{
		
		v_tmp  = _mm256_load_pd( &ptr_t[0] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[0] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[0], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[N][ii] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[N][0+(ii+0)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[N][1+(ii+1)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[N][2*ii+0] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[N][ii] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[N][(ii+0)*bs], u_bl );
		_mm_storeh_pd( &pl[N][(ii+1)*bs], u_bl );
		_mm_store_pd( &pl2[N][ii+0], u_bl );

		v_tmp  = _mm256_load_pd( &ptr_t[4] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[4], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[4] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[4], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[4], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[N][ii+2] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[N][2+(ii+2)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[N][3+(ii+3)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[N][2*ii+4] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[N][ii+2] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[N][(ii+2)*bs], u_bl );
		_mm_storeh_pd( &pl[N][(ii+3)*bs], u_bl );
		_mm_store_pd( &pl2[N][ii+2], u_bl );


		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;

		}
	if(ii<kmax)
		{

/*		bs0 = nb-ii;*/
		bs0 = kmax-ii;
		ll = 0;
		
		if(bs0>=2)
			{

			v_tmp  = _mm256_load_pd( &ptr_t[0] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[0], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[N][ii] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[N][0+(ii+0)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[N][1+(ii+1)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[N][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[N][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[N][(ii+0)*bs], u_bl );
			_mm_storeh_pd( &pl[N][(ii+1)*bs], u_bl );
			_mm_store_pd( &pl2[N][ii+0], u_bl );

			ptr_t     += 4;
			ptr_lam   += 4;
			ptr_lamt  += 4;
			ptr_dlam  += 4;
			ptr_t_inv += 4;
			
			ll   += 2;
			bs0  -= 2;

			}
		
		if(bs0>0)
			{
			
			u_tmp  = _mm_load_pd( &ptr_t[0] );
			u_tmp  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp );
			_mm_store_pd( &ptr_t_inv[0], u_tmp ); // store t_inv
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );
			u_tmp  = _mm_hadd_pd( u_lamt, u_lamt ); // [ lamt[0]+lamt[1] , xxx ]
			u_bd   = _mm_load_sd( &bd[N][ii+ll] );
			u_bd   = _mm_add_sd( u_bd, u_tmp );
			_mm_store_sd( &pd[N][ll+(ii+ll)*bs+ii*cnz], u_bd );
			u_db   = _mm_load_pd( &db[N][2*ii+2*ll+0] );
			u_db   = _mm_mul_pd( u_db, u_lamt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_add_pd( u_lam, u_db );
			u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
			u_bl   = _mm_load_sd( &bl[N][ii+ll] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_sd( &pl[N][(ii+ll)*bs], u_bl );
			_mm_store_pd( &pl2[N][ii+ll], u_bl );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/

			}
		}

	
	return;

	}



void d_compute_alpha_box_mpc(int N, int k0, int k1, int kmax, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db)
	{
	
	__m128
		s_sign, s_ones, s_mask0, s_mask1, s_zeros,
		s_lam, s_dlam, s_t, s_dt, s_tmp0, s_tmp1, s_alpha;
	
	__m256d
		v_sign, v_temp,
		v_dt, v_dux, v_db, v_dlam, v_lamt, v_t, v_alpha, v_lam;
	
	__m128d
		u_sign,	u_temp,
		u_dt, u_dux, u_db, u_dlam, u_lamt, u_t, u_alpha;
	
	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );
	u_sign = _mm_loaddup_pd( (double *) &long_sign );

	int int_sign = 0x80000000;
	s_sign = _mm_broadcast_ss( (float *) &int_sign );
	
	s_ones  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_zeros = _mm_setzero_ps( );

	s_alpha  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );



	const int bs = 4; //d_get_mr();
	
	double alpha = ptr_alpha[0];
	
	int kna = ((k1+bs-1)/bs)*bs;

	int jj, ll;


	// first stage
	ll = 0;
	for(; ll<k0-3; ll+=4) // TODO avx single prec
		{

		v_db    = _mm256_load_pd( &db[0][ll] );
		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[0][ll/2+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[0][ll/2+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_lamt  = _mm256_load_pd( &lamt[0][ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dt );
		v_dlam  = _mm256_load_pd( &dlam[0][ll] );
		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
		_mm256_store_pd( &dlam[0][ll], v_dlam );
		v_t     = _mm256_load_pd( &t[0][ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[0][ll], v_dt );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_dt    = _mm256_cvtpd_ps( v_dt );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		s_mask1 = _mm_cmplt_ps( s_dt, s_zeros );
		s_lam   = _mm256_cvtpd_ps( _mm256_load_pd( &lam[0][ll] ) );
		s_t     = _mm256_cvtpd_ps( v_t );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_t     = _mm_xor_ps( s_t, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp1  = _mm_div_ps( s_t, s_dt );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_tmp1  = _mm_blendv_ps( s_ones, s_tmp1, s_mask1 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp1 );

		}

/*	v_alpha = _mm256_cvtps_pd( s_alpha );*/
/*	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );*/
/*	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );*/
/*	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );*/
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
/*	_mm_store_sd( &alpha, u_alpha );*/

	for(; ll<k0; ll+=2)
		{

		u_db    = _mm_load_pd( &db[0][ll] );
		u_dux   = _mm_loaddup_pd( &dux[0][ll/2+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_lamt  = _mm_load_pd( &lamt[0][ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[0][ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[0][ll], u_dlam );
		u_t     = _mm_load_pd( &t[0][ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[0][ll], u_dt );
		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( _mm_load_pd( &lam[0][ll] ) ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

/*		dt[0][ll+0] =   dux[0][ll/2] - db[0][ll+0];*/
/*		dt[0][ll+1] = - dux[0][ll/2] - db[0][ll+1];*/
/*		dlam[0][ll+0] -= lamt[0][ll+0] * dt[0][ll+0];*/
/*		dlam[0][ll+1] -= lamt[0][ll+1] * dt[0][ll+1];*/
/*		if( -alpha*dlam[0][ll+0]>lam[0][ll+0] )*/
/*			{*/
/*			alpha = - lam[0][ll+0] / dlam[0][ll+0];*/
/*			}*/
/*		if( -alpha*dlam[0][ll+1]>lam[0][ll+1] )*/
/*			{*/
/*			alpha = - lam[0][ll+1] / dlam[0][ll+1];*/
/*			}*/
/*		dt[0][ll+0] -= t[0][ll+0];*/
/*		dt[0][ll+1] -= t[0][ll+1];*/
/*		if( -alpha*dt[0][ll+0]>t[0][ll+0] )*/
/*			{*/
/*			alpha = - t[0][ll+0] / dt[0][ll+0];*/
/*			}*/
/*		if( -alpha*dt[0][ll+1]>t[0][ll+1] )*/
/*			{*/
/*			alpha = - t[0][ll+1] / dt[0][ll+1];*/
/*			}*/

		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{

		ll = 0;
		for(; ll<kmax-3; ll+=4)
			{

			v_db    = _mm256_load_pd( &db[jj][ll] );
			v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll/2+0] ) );
			v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll/2+1] ) );
			v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
			v_dt    = _mm256_addsub_pd( v_db, v_dux );
			v_dt    = _mm256_xor_pd( v_dt, v_sign );
			v_lamt  = _mm256_load_pd( &lamt[jj][ll] );
			v_temp  = _mm256_mul_pd( v_lamt, v_dt );
			v_dlam  = _mm256_load_pd( &dlam[jj][ll] );
			v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
			_mm256_store_pd( &dlam[jj][ll], v_dlam );
			v_t     = _mm256_load_pd( &t[jj][ll] );
			v_dt    = _mm256_sub_pd( v_dt, v_t );
			_mm256_store_pd( &dt[jj][ll], v_dt );
			s_dlam  = _mm256_cvtpd_ps( v_dlam );
			s_dt    = _mm256_cvtpd_ps( v_dt );
			s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
			s_mask1 = _mm_cmplt_ps( s_dt, s_zeros );
			s_lam   = _mm256_cvtpd_ps( _mm256_load_pd( &lam[jj][ll] ) );
			s_t     = _mm256_cvtpd_ps( v_t );
			s_lam   = _mm_xor_ps( s_lam, s_sign );
			s_t     = _mm_xor_ps( s_t, s_sign );
			s_tmp0  = _mm_div_ps( s_lam, s_dlam );
			s_tmp1  = _mm_div_ps( s_t, s_dt );
			s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
			s_tmp1  = _mm_blendv_ps( s_ones, s_tmp1, s_mask1 );
			s_alpha = _mm_min_ps( s_alpha, s_tmp0 );
			s_alpha = _mm_min_ps( s_alpha, s_tmp1 );
			}

/*		v_alpha = _mm256_cvtps_pd( s_alpha );*/
/*		u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );*/
/*		u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );*/
/*		u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );*/
/*		u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
/*		_mm_store_sd( &alpha, u_alpha );*/

		for(; ll<kmax; ll+=2)
			{

			u_db    = _mm_load_pd( &db[jj][ll] );
			u_dux   = _mm_loaddup_pd( &dux[jj][ll/2+0] );
			u_dt    = _mm_addsub_pd( u_db, u_dux );
			u_dt    = _mm_xor_pd( u_dt, u_sign );
			u_lamt  = _mm_load_pd( &lamt[jj][ll] );
			u_temp  = _mm_mul_pd( u_lamt, u_dt );
			u_dlam  = _mm_load_pd( &dlam[jj][ll] );
			u_dlam  = _mm_sub_pd( u_dlam, u_temp );
			_mm_store_pd( &dlam[jj][ll], u_dlam );
			u_t     = _mm_load_pd( &t[jj][ll] );
			u_dt    = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][ll], u_dt );
			v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
			s_dlam  = _mm256_cvtpd_ps( v_dlam );
			s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
			v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( _mm_load_pd( &lam[jj][ll] ) ), _mm256_castpd128_pd256( u_t ), 0x20 );
			s_lam   = _mm256_cvtpd_ps( v_lam );
			s_lam   = _mm_xor_ps( s_lam, s_sign );
			s_tmp0  = _mm_div_ps( s_lam, s_dlam );
			s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
			s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

/*			dt[jj][ll+0] =   dux[jj][ll/2] - db[jj][ll+0];*/
/*			dt[jj][ll+1] = - dux[jj][ll/2] - db[jj][ll+1];*/
/*			dlam[jj][ll+0] -= lamt[jj][ll+0] * dt[jj][ll+0];*/
/*			dlam[jj][ll+1] -= lamt[jj][ll+1] * dt[jj][ll+1];*/
/*			if( -alpha*dlam[jj][ll+0]>lam[jj][ll+0] )*/
/*				{*/
/*				alpha = - lam[jj][ll+0] / dlam[jj][ll+0];*/
/*				}*/
/*			if( -alpha*dlam[jj][ll+1]>lam[jj][ll+1] )*/
/*				{*/
/*				alpha = - lam[jj][ll+1] / dlam[jj][ll+1];*/
/*				}*/
/*			dt[jj][ll+0] -= t[jj][ll+0];*/
/*			dt[jj][ll+1] -= t[jj][ll+1];*/
/*			if( -alpha*dt[jj][ll+0]>t[jj][ll+0] )*/
/*				{*/
/*				alpha = - t[jj][ll+0] / dt[jj][ll+0];*/
/*				}*/
/*			if( -alpha*dt[jj][ll+1]>t[jj][ll+1] )*/
/*				{*/
/*				alpha = - t[jj][ll+1] / dt[jj][ll+1];*/
/*				}*/

			}

		}		

	// last stage
	ll = k1;
	for(; ll<kna; ll+=2)
		{

		u_db    = _mm_load_pd( &db[N][ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll/2+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_lamt  = _mm_load_pd( &lamt[N][ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[N][ll], u_dlam );
		u_t     = _mm_load_pd( &t[N][ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][ll], u_dt );
		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( _mm_load_pd( &lam[N][ll] ) ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

/*		dt[N][ll+0] =   dux[N][ll/2] - db[N][ll+0];*/
/*		dt[N][ll+1] = - dux[N][ll/2] - db[N][ll+1];*/
/*		dlam[N][ll+0] -= lamt[N][ll+0] * dt[N][ll+0];*/
/*		dlam[N][ll+1] -= lamt[N][ll+1] * dt[N][ll+1];*/
/*		if( -alpha*dlam[N][ll+0]>lam[N][ll+0] )*/
/*			{*/
/*			alpha = - lam[N][ll+0] / dlam[N][ll+0];*/
/*			}*/
/*		if( -alpha*dlam[N][ll+1]>lam[N][ll+1] )*/
/*			{*/
/*			alpha = - lam[N][ll+1] / dlam[N][ll+1];*/
/*			}*/
/*		dt[N][ll+0] -= t[N][ll+0];*/
/*		dt[N][ll+1] -= t[N][ll+1];*/
/*		if( -alpha*dt[N][ll+0]>t[N][ll+0] )*/
/*			{*/
/*			alpha = - t[N][ll+0] / dt[N][ll+0];*/
/*			}*/
/*		if( -alpha*dt[N][ll+1]>t[N][ll+1] )*/
/*			{*/
/*			alpha = - t[N][ll+1] / dt[N][ll+1];*/
/*			}*/

		}
		
	for(; ll<kmax-3; ll+=4)
		{

		v_db    = _mm256_load_pd( &db[N][ll] );
		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[N][ll/2+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[N][ll/2+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_lamt  = _mm256_load_pd( &lamt[N][ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dt );
		v_dlam  = _mm256_load_pd( &dlam[N][ll] );
		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
		_mm256_store_pd( &dlam[N][ll], v_dlam );
		v_t     = _mm256_load_pd( &t[N][ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[N][ll], v_dt );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_dt    = _mm256_cvtpd_ps( v_dt );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		s_mask1 = _mm_cmplt_ps( s_dt, s_zeros );
		s_lam   = _mm256_cvtpd_ps( _mm256_load_pd( &lam[N][ll] ) );
		s_t     = _mm256_cvtpd_ps( v_t );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_t     = _mm_xor_ps( s_t, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp1  = _mm_div_ps( s_t, s_dt );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_tmp1  = _mm_blendv_ps( s_ones, s_tmp1, s_mask1 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp1 );

		}

/*	v_alpha = _mm256_cvtps_pd( s_alpha );*/
/*	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );*/
/*	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );*/
/*	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );*/
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
/*	_mm_store_sd( &alpha, u_alpha );*/

	for(; ll<kmax; ll+=2)
		{

		u_db    = _mm_load_pd( &db[N][ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll/2+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_lamt  = _mm_load_pd( &lamt[N][ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[N][ll], u_dlam );
		u_t     = _mm_load_pd( &t[N][ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][ll], u_dt );
		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( _mm_load_pd( &lam[N][ll] ) ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

/*		dt[N][ll+0] =   dux[N][ll/2] - db[N][ll+0];*/
/*		dt[N][ll+1] = - dux[N][ll/2] - db[N][ll+1];*/
/*		dlam[N][ll+0] -= lamt[N][ll+0] * dt[N][ll+0];*/
/*		dlam[N][ll+1] -= lamt[N][ll+1] * dt[N][ll+1];*/
/*		if( -alpha*dlam[N][ll+0]>lam[N][ll+0] )*/
/*			{*/
/*			alpha = - lam[N][ll+0] / dlam[N][ll+0];*/
/*			}*/
/*		if( -alpha*dlam[N][ll+1]>lam[N][ll+1] )*/
/*			{*/
/*			alpha = - lam[N][ll+1] / dlam[N][ll+1];*/
/*			}*/
/*		dt[N][ll+0] -= t[N][ll+0];*/
/*		dt[N][ll+1] -= t[N][ll+1];*/
/*		if( -alpha*dt[N][ll+0]>t[N][ll+0] )*/
/*			{*/
/*			alpha = - t[N][ll+0] / dt[N][ll+0];*/
/*			}*/
/*		if( -alpha*dt[N][ll+1]>t[N][ll+1] )*/
/*			{*/
/*			alpha = - t[N][ll+1] / dt[N][ll+1];*/
/*			}*/

		}

	v_alpha = _mm256_cvtps_pd( s_alpha );
	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );
	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );
	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
	_mm_store_sd( &alpha, u_alpha );


/*	ll = k0;*/
/*	for(; ll<kna; ll+=2)*/
/*		{*/
/*		dt[ll+0] =   dux[ll/2] - db[ll+0];*/
/*		dt[ll+1] = - dux[ll/2] - db[ll+1];*/
/*		dlam[ll+0] -= lamt[ll+0] * dt[ll+0];*/
/*		dlam[ll+1] -= lamt[ll+1] * dt[ll+1];*/
/*		if( -alpha*dlam[ll+0]>lam[ll+0] )*/
/*			{*/
/*			alpha = - lam[ll+0] / dlam[ll+0];*/
/*			}*/
/*		if( -alpha*dlam[ll+1]>lam[ll+1] )*/
/*			{*/
/*			alpha = - lam[ll+1] / dlam[ll+1];*/
/*			}*/
/*		dt[ll+0] -= t[ll+0];*/
/*		dt[ll+1] -= t[ll+1];*/
/*		if( -alpha*dt[ll+0]>t[ll+0] )*/
/*			{*/
/*			alpha = - t[ll+0] / dt[ll+0];*/
/*			}*/
/*		if( -alpha*dt[ll+1]>t[ll+1] )*/
/*			{*/
/*			alpha = - t[ll+1] / dt[ll+1];*/
/*			}*/
/*		}*/
/*		*/
/*	for(; ll<kmax-3; ll+=4)*/
/*		{*/

/*		v_db    = _mm256_load_pd( &db[ll] );*/
/*		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[ll/2+0] ) );*/
/*		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[ll/2+1] ) );*/
/*		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );*/
/*		v_dt    = _mm256_addsub_pd( v_db, v_dux );*/
/*		v_dt    = _mm256_xor_pd( v_dt, v_sign );*/
/*		v_lamt  = _mm256_load_pd( &lamt[ll] );*/
/*		v_temp  = _mm256_mul_pd( v_lamt, v_dt );*/
/*		v_dlam  = _mm256_load_pd( &dlam[ll] );*/
/*		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );*/
/*		_mm256_store_pd( &dlam[ll], v_dlam );*/
/*		v_t     = _mm256_load_pd( &t[ll] );*/
/*		v_dt    = _mm256_sub_pd( v_dt, v_t );*/
/*		_mm256_store_pd( &dt[ll], v_dt );*/
/*		s_dlam  = _mm256_cvtpd_ps( v_dlam );*/
/*		s_dt    = _mm256_cvtpd_ps( v_dt );*/
/*		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );*/
/*		s_mask1 = _mm_cmplt_ps( s_dt, s_zeros );*/
/*		s_lam   = _mm256_cvtpd_ps( _mm256_load_pd( &lam[ll] ) );*/
/*		s_t     = _mm256_cvtpd_ps( v_t );*/
/*		s_lam   = _mm_xor_ps( s_lam, s_sign );*/
/*		s_t     = _mm_xor_ps( s_t, s_sign );*/
/*		s_tmp0  = _mm_div_ps( s_lam, s_dlam );*/
/*		s_tmp1  = _mm_div_ps( s_t, s_dt );*/
/*		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );*/
/*		s_tmp1  = _mm_blendv_ps( s_ones, s_tmp1, s_mask1 );*/
/*		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );*/
/*		s_alpha = _mm_min_ps( s_alpha, s_tmp1 );*/
/*		}*/

/*	v_alpha = _mm256_cvtps_pd( s_alpha );*/
/*	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );*/
/*	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );*/
/*	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );*/
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
/*	_mm_store_sd( &alpha, u_alpha );*/

/*	for(; ll<kmax; ll+=2)*/
/*		{*/
/*		dt[ll+0] =   dux[ll/2] - db[ll+0];*/
/*		dt[ll+1] = - dux[ll/2] - db[ll+1];*/
/*		dlam[ll+0] -= lamt[ll+0] * dt[ll+0];*/
/*		dlam[ll+1] -= lamt[ll+1] * dt[ll+1];*/
/*		if( -alpha*dlam[ll+0]>lam[ll+0] )*/
/*			{*/
/*			alpha = - lam[ll+0] / dlam[ll+0];*/
/*			}*/
/*		if( -alpha*dlam[ll+1]>lam[ll+1] )*/
/*			{*/
/*			alpha = - lam[ll+1] / dlam[ll+1];*/
/*			}*/
/*		dt[ll+0] -= t[ll+0];*/
/*		dt[ll+1] -= t[ll+1];*/
/*		if( -alpha*dt[ll+0]>t[ll+0] )*/
/*			{*/
/*			alpha = - t[ll+0] / dt[ll+0];*/
/*			}*/
/*		if( -alpha*dt[ll+1]>t[ll+1] )*/
/*			{*/
/*			alpha = - t[ll+1] / dt[ll+1];*/
/*			}*/
/*		}*/
	
	ptr_alpha[0] = alpha;

	return;
	
	}


void d_update_var_mpc(int nx, int nu, int N, int nb, int nbu, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{
	
	int jj, ll;
	
	__m128d
		u_ux, u_dux, u_pi, u_dpi, u_t, u_dt, u_lam, u_dlam, u_mu, u_tmp;

	__m256d
		v_alpha, v_ux, v_dux, v_pi, v_dpi, v_t, v_dt, v_lam, v_dlam, v_mu;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_mu = _mm256_setzero_pd();
	u_mu = _mm_setzero_pd();


	// first stage
	jj = 0;
	
	// update inputs
	ll = 0;
	for(; ll<nu-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		}
	if(ll<nu-1)
		{
		u_ux  = _mm_load_pd( &ux[jj][ll] );
		u_dux = _mm_load_pd( &dux[jj][ll] );
		u_dux = _mm_sub_pd( u_dux, u_ux );
		u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_pd( u_ux, u_dux );
		_mm_store_pd( &ux[jj][ll], u_ux );
		ll += 2;
		}
	if(ll<nu)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		}

	// box constraints
	ll = 0;
	for(; ll<2*nbu-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		_mm256_store_pd( &t[jj][ll], v_t );
		_mm256_store_pd( &lam[jj][ll], v_lam );
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nbu-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][ll], u_t );
		_mm_store_pd( &lam[jj][ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	// middle stage
	for(jj=1; jj<N; jj++)
		{
		
		ll = 0;
		for(; ll<nx+nu-3; ll+=4)
			{
			v_ux  = _mm256_load_pd( &ux[jj][ll] );
			v_dux = _mm256_load_pd( &dux[jj][ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
			_mm256_store_pd( &ux[jj][ll], v_ux );
			}
		if(ll<nx+nu-1)
			{
			u_ux  = _mm_load_pd( &ux[jj][ll] );
			u_dux = _mm_load_pd( &dux[jj][ll] );
			u_dux = _mm_sub_pd( u_dux, u_ux );
			u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
			u_ux  = _mm_add_pd( u_ux, u_dux );
			_mm_store_pd( &ux[jj][ll], u_ux );
			ll += 2;
			}
		if(ll<nx+nu)
			{
			u_ux  = _mm_load_sd( &ux[jj][ll] );
			u_dux = _mm_load_sd( &dux[jj][ll] );
			u_dux = _mm_sub_sd( u_dux, u_ux );
			u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
			u_ux  = _mm_add_sd( u_ux, u_dux );
			_mm_store_sd( &ux[jj][ll], u_ux );
			}
		
		ll = 0;
		for(; ll<nx-3; ll+=4)
			{
			v_pi  = _mm256_load_pd( &pi[jj][ll] );
			v_dpi = _mm256_load_pd( &dpi[jj][ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
			_mm256_store_pd( &pi[jj][ll], v_pi );
			}
		if(ll<nx-1)
			{
			u_pi  = _mm_load_pd( &pi[jj][ll] );
			u_dpi = _mm_load_pd( &dpi[jj][ll] );
			u_dpi = _mm_sub_pd( u_dpi, u_pi );
			u_dpi = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
			u_pi  = _mm_add_pd( u_pi, u_dpi );
			_mm_store_pd( &pi[jj][ll], u_pi );
			ll += 2;
			}
		if(ll<nx)
			{
			u_pi  = _mm_load_sd( &pi[jj][ll] );
			u_dpi = _mm_load_sd( &dpi[jj][ll] );
			u_dpi = _mm_sub_sd( u_dpi, u_pi );
			u_dpi = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
			u_pi  = _mm_add_sd( u_pi, u_dpi );
			_mm_store_sd( &pi[jj][ll], u_pi );
			}

		ll = 0;
		for(; ll<2*nb-3; ll+=4)
			{
			v_t    = _mm256_load_pd( &t[jj][ll] );
			v_lam  = _mm256_load_pd( &lam[jj][ll] );
			v_dt   = _mm256_load_pd( &dt[jj][ll] );
			v_dlam = _mm256_load_pd( &dlam[jj][ll] );
			v_dt   = _mm256_mul_pd( v_alpha, v_dt );
			v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
			v_t    = _mm256_add_pd( v_t, v_dt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			_mm256_store_pd( &t[jj][ll], v_t );
			_mm256_store_pd( &lam[jj][ll], v_lam );
			v_lam  = _mm256_mul_pd( v_lam, v_t );
			v_mu   = _mm256_add_pd( v_mu, v_lam );
			}
		if(ll<2*nb-1)
			{
			u_t    = _mm_load_pd( &t[jj][ll] );
			u_lam  = _mm_load_pd( &lam[jj][ll] );
			u_dt   = _mm_load_pd( &dt[jj][ll] );
			u_dlam = _mm_load_pd( &dlam[jj][ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			_mm_store_pd( &t[jj][ll], u_t );
			_mm_store_pd( &lam[jj][ll], u_lam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		}

	// last stage
	jj = N;
	
	// update states
	ll = nu;
	if(nu%2==1)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		ll++;
		}
	if((4-nu%4)%4>1)
		{
		u_ux  = _mm_load_pd( &ux[jj][ll] );
		u_dux = _mm_load_pd( &dux[jj][ll] );
		u_dux = _mm_sub_pd( u_dux, u_ux );
		u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_pd( u_ux, u_dux );
		_mm_store_pd( &ux[jj][ll], u_ux );
		ll += 2;
		}
	for(; ll<nx+nu-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		}
	if(ll<nx+nu-1)
		{
		u_ux  = _mm_load_pd( &ux[jj][ll] );
		u_dux = _mm_load_pd( &dux[jj][ll] );
		u_dux = _mm_sub_pd( u_dux, u_ux );
		u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_pd( u_ux, u_dux );
		_mm_store_pd( &ux[jj][ll], u_ux );
		ll += 2;
		}
	if(ll<nx+nu)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		}

	ll = 0;
	for(; ll<nx-3; ll+=4)
		{
		v_pi  = _mm256_load_pd( &pi[jj][ll] );
		v_dpi = _mm256_load_pd( &dpi[jj][ll] );
		v_dpi = _mm256_sub_pd( v_dpi, v_pi );
		v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
		v_pi  = _mm256_add_pd( v_pi, v_dpi );
		_mm256_store_pd( &pi[jj][ll], v_pi );
		}
	if(ll<nx-1)
		{
		u_pi  = _mm_load_pd( &pi[jj][ll] );
		u_dpi = _mm_load_pd( &dpi[jj][ll] );
		u_dpi = _mm_sub_pd( u_dpi, u_pi );
		u_dpi = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
		u_pi  = _mm_add_pd( u_pi, u_dpi );
		_mm_store_pd( &pi[jj][ll], u_pi );
		ll += 2;
		}
	if(ll<nx)
		{
		u_pi  = _mm_load_sd( &pi[jj][ll] );
		u_dpi = _mm_load_sd( &dpi[jj][ll] );
		u_dpi = _mm_sub_sd( u_dpi, u_pi );
		u_dpi = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
		u_pi  = _mm_add_sd( u_pi, u_dpi );
		_mm_store_sd( &pi[jj][ll], u_pi );
		}

	ll = 2*nu;
	if(nu%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][ll], u_t );
		_mm_store_pd( &lam[jj][ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		ll += 2;
		}
	for(; ll<2*nb-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		_mm256_store_pd( &t[jj][ll], v_t );
		_mm256_store_pd( &lam[jj][ll], v_lam );
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nb-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][ll], u_t );
		_mm_store_pd( &lam[jj][ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	u_tmp = _mm256_extractf128_pd( v_mu, 0x1 );
	u_mu  = _mm_add_pd( u_mu, _mm256_castpd256_pd128( v_mu ) );
	u_mu  = _mm_add_pd( u_mu, u_tmp );
	u_mu  = _mm_hadd_pd( u_mu, u_mu );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu  = _mm_mul_sd( u_mu, u_tmp );
	_mm_store_sd( ptr_mu, u_mu );
		

	return;
	
	}


void d_compute_mu_mpc(int N, int nbu, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{
	
	int jj, ll;
	
	__m128d
		u_ux, u_dux, u_pi, u_dpi, u_t, u_dt, u_lam, u_dlam, u_mu, u_tmp;

	__m256d
		v_alpha, v_ux, v_dux, v_pi, v_dpi, v_t, v_dt, v_lam, v_dlam, v_mu;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_mu = _mm256_setzero_pd();
	u_mu = _mm_setzero_pd();


	// first stage
	jj = 0;
	
	// box constraints
	ll = 0;
	for(; ll<2*nbu-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
/*		_mm256_store_pd( &t[jj][ll], v_t );*/
/*		_mm256_store_pd( &lam[jj][ll], v_lam );*/
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nbu-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
/*		_mm_store_pd( &t[jj][ll], u_t );*/
/*		_mm_store_pd( &lam[jj][ll], u_lam );*/
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	// middle stage
	for(jj=1; jj<N; jj++)
		{
		
		ll = 0;
		for(; ll<2*nb-3; ll+=4)
			{
			v_t    = _mm256_load_pd( &t[jj][ll] );
			v_lam  = _mm256_load_pd( &lam[jj][ll] );
			v_dt   = _mm256_load_pd( &dt[jj][ll] );
			v_dlam = _mm256_load_pd( &dlam[jj][ll] );
			v_dt   = _mm256_mul_pd( v_alpha, v_dt );
			v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
			v_t    = _mm256_add_pd( v_t, v_dt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
/*			_mm256_store_pd( &t[jj][ll], v_t );*/
/*			_mm256_store_pd( &lam[jj][ll], v_lam );*/
			v_lam  = _mm256_mul_pd( v_lam, v_t );
			v_mu   = _mm256_add_pd( v_mu, v_lam );
			}
		if(ll<2*nb-1)
			{
			u_t    = _mm_load_pd( &t[jj][ll] );
			u_lam  = _mm_load_pd( &lam[jj][ll] );
			u_dt   = _mm_load_pd( &dt[jj][ll] );
			u_dlam = _mm_load_pd( &dlam[jj][ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
/*			_mm_store_pd( &t[jj][ll], u_t );*/
/*			_mm_store_pd( &lam[jj][ll], u_lam );*/
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		}

	// last stage
	jj = N;
	
	ll = 2*nu;
	if(nu%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
/*		_mm_store_pd( &t[jj][ll], u_t );*/
/*		_mm_store_pd( &lam[jj][ll], u_lam );*/
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		ll += 2;
		}
	for(; ll<2*nb-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
/*		_mm256_store_pd( &t[jj][ll], v_t );*/
/*		_mm256_store_pd( &lam[jj][ll], v_lam );*/
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nb-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
/*		_mm_store_pd( &t[jj][ll], u_t );*/
/*		_mm_store_pd( &lam[jj][ll], u_lam );*/
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	u_tmp = _mm256_extractf128_pd( v_mu, 0x1 );
	u_mu  = _mm_add_pd( u_mu, _mm256_castpd256_pd128( v_mu ) );
	u_mu  = _mm_add_pd( u_mu, u_tmp );
	u_mu  = _mm_hadd_pd( u_mu, u_mu );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu  = _mm_mul_sd( u_mu, u_tmp );
	_mm_store_sd( ptr_mu, u_mu );
		

	return;

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











/*		mu = 0;*/
/*		// update inputs*/
/*		for(ll=0; ll<nu; ll++)*/
/*			ux[0][ll] += alpha*(dux[0][ll] - ux[0][ll]);*/
/*		// box constraints*/
/*		for(ll=0; ll<2*nbu; ll+=2)*/
/*			{*/
/*			lam[0][ll+0] += alpha*dlam[0][ll+0];*/
/*			lam[0][ll+1] += alpha*dlam[0][ll+1];*/
/*			t[0][ll+0] += alpha*dt[0][ll+0];*/
/*			t[0][ll+1] += alpha*dt[0][ll+1];*/
/*			mu += lam[0][ll+0] * t[0][ll+0] + lam[0][ll+1] * t[0][ll+1];*/
/*			}*/
/*		for(jj=1; jj<N; jj++)*/
/*			{*/
/*			// update inputs*/
/*			for(ll=0; ll<nu; ll++)*/
/*				ux[jj][ll] += alpha*(dux[jj][ll] - ux[jj][ll]);*/
/*			// update states*/
/*			for(ll=0; ll<nx; ll++)*/
/*				ux[jj][nu+ll] += alpha*(dux[jj][nu+ll] - ux[jj][nu+ll]);*/
/*			// update equality constrained multipliers*/
/*			for(ll=0; ll<nx; ll++)*/
/*				pi[jj][ll] += alpha*(dpi[jj][ll] - pi[jj][ll]);*/
/*			// box constraints*/
/*			for(ll=0; ll<2*nb; ll+=2)*/
/*				{*/
/*				lam[jj][ll+0] += alpha*dlam[jj][ll+0];*/
/*				lam[jj][ll+1] += alpha*dlam[jj][ll+1];*/
/*				t[jj][ll+0] += alpha*dt[jj][ll+0];*/
/*				t[jj][ll+1] += alpha*dt[jj][ll+1];*/
/*				mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1];*/
/*				}*/
/*			}*/
/*		// update states*/
/*		for(ll=0; ll<nx; ll++)*/
/*			ux[N][nu+ll] += alpha*(dux[N][nu+ll] - ux[N][nu+ll]);*/
/*		// update equality constrained multipliers*/
/*		for(ll=0; ll<nx; ll++)*/
/*			pi[N][ll] += alpha*(dpi[N][ll] - pi[N][ll]);*/
/*		// box constraints*/
/*		for(ll=2*nu; ll<2*nb; ll+=2)*/
/*			{*/
/*			lam[N][ll+0] += alpha*dlam[N][ll+0];*/
/*			lam[N][ll+1] += alpha*dlam[N][ll+1];*/
/*			t[N][ll+0] += alpha*dt[N][ll+0];*/
/*			t[N][ll+1] += alpha*dt[N][ll+1];*/
/*			mu += lam[N][ll+0] * t[N][ll+0] + lam[N][ll+1] * t[N][ll+1];*/
/*			}*/
/*		mu *= mu_scal;*/

