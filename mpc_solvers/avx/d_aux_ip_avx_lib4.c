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

#include <math.h> // TODO remove if not needed

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX

#include "../../include/blas_d.h"
#include "../../include/block_size.h"



void d_init_var_hard_mpc_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **ux, double **pi, double **pDCt, double **db, double **t, double **lam, double mu0, int warm_start)
	{

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	int jj, ll, ii;

	int nb0, pnb, ng0, png, cng;
	
	double
		*ptr_t, *ptr_lam, *ptr_db;

	double thr0 = 0.1; // minimum vale of t (minimum distance from a constraint)


	// cold start
	if(warm_start==0)
		{
		for(ll=0; ll<nu[jj]; ll++)
			{
			ux[jj][ll] = 0.0;
			}
		for(jj=1; jj<=N; jj++)
			{
			for(ll=0; ll<nu[jj]+nx[jj]; ll++)
				{
				ux[jj][ll] = 0.0;
				}
			}
		}


	// check bounds & initialize multipliers
	for(jj=0; jj<=N; jj++)
		{
		nb0 = nb[jj];
		pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints
		for(ll=0; ll<nb0; ll++)
			{
			t[jj][ll]     = - db[jj][ll]     + ux[jj][idxb[jj][ll]];
			t[jj][pnb+ll] = - db[jj][pnb+ll] - ux[jj][idxb[jj][ll]];
			if(t[jj][ll] < thr0)
				{
				if(t[jj][pnb+ll] < thr0)
					{
					ux[jj][idxb[jj][ll]] = ( - db[jj][pnb+ll] + db[jj][ll])*0.5;
					t[jj][ll]     = - db[jj][ll]     + ux[jj][idxb[jj][ll]];
					t[jj][pnb+ll] = - db[jj][pnb+ll] - ux[jj][idxb[jj][ll]];
					}
				else
					{
					t[jj][ll] = thr0;
					ux[jj][idxb[jj][ll]] = db[jj][ll] + thr0;
					}
				}
			else if(t[jj][pnb+ll] < thr0)
				{
				t[jj][pnb+ll] = thr0;
				ux[jj][idxb[jj][ll]] = - db[jj][pnb+ll] - thr0;
				}
			lam[jj][ll]     = mu0/t[jj][ll];
			lam[jj][pnb+ll] = mu0/t[jj][pnb+ll];
			}
		}


	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx[jj]; ll++)
			pi[jj][ll] = 0.0; // initialize multipliers to zero


	// TODO find a better way to initialize general constraints
	for(jj=0; jj<=N; jj++)
		{
		ng0 = ng[jj];
		png = (ng0+bs-1)/bs*bs;
		cng = (ng0+ncl-1)/ncl*ncl;
		if(ng0>0)
			{
			ptr_t   = t[jj];
			ptr_lam = lam[jj];
			ptr_db  = db[jj];
			dgemv_t_lib(nu[jj]+nx[jj], ng0, pDCt[jj], cng, ux[jj], 0, ptr_t+2*pnb, ptr_t+2*pnb);
			for(ll=2*pnb; ll<2*pnb+ng0; ll++)
				{
				ptr_t[ll+png] = - ptr_t[ll];
				ptr_t[ll]      += - ptr_db[ll];
				ptr_t[ll+png] += - ptr_db[ll+png];
				ptr_t[ll]      = fmax( thr0, ptr_t[ll] );
				ptr_t[png+ll] = fmax( thr0, ptr_t[png+ll] );
				ptr_lam[ll]      = mu0/ptr_t[ll];
				ptr_lam[png+ll] = mu0/ptr_t[png+ll];
				}
			}
		}

	}



void d_init_var_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, double **ux, double **pi, double **pDCt, double **db, double **t, double **lam, double mu0, int warm_start)
	{

	const int nbu = nu<nb ? nu : nb ;
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	//const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box constraints
	const int pnb  = bs*((nb+bs-1)/bs); // simd aligned number of box constraints
	//const int ang = nal*((ng+nal-1)/nal); // cache aligned number of general constraints
	const int png  = bs*((ng+bs-1)/bs); // simd aligned number of general constraints
	const int pngN = bs*((ngN+bs-1)/bs); // simd aligned number of general constraints at stage N
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);

	int jj, ll, ii;

	double
		*ptr_t, *ptr_lam, *ptr_db;

	double thr0 = 0.1; // minimum value of t (minimum distance from a constraint)

	if(warm_start==1)
		{
		for(ll=0; ll<nbu; ll++)
			{
			t[0][ll]     =  - db[0][ll]     + ux[0][ll];
			t[0][pnb+ll] =  - db[0][pnb+ll] - ux[0][ll];
			if(t[0][ll] < thr0)
				{
				if(t[0][pnb+ll] < thr0)
					{
					ux[0][ll] = ( - db[0][pnb+ll] + db[0][ll])*0.5;
					t[0][ll]     = - db[0][ll]     + ux[0][ll] ;
					t[0][pnb+ll] = - db[0][pnb+ll] - ux[0][ll];
					}
				else
					{
					t[0][ll] = thr0;
					ux[0][ll] = db[0][ll] + thr0;
					}
				}
			else if(t[0][pnb+ll] < thr0)
				{
				t[0][pnb+ll] = thr0;
				ux[0][ll] = - db[0][pnb+ll] - thr0;
				}
			lam[0][ll]     = mu0/t[0][ll];
			lam[0][pnb+ll] = mu0/t[0][pnb+ll];
			}
		for(; ll<nb; ll++)
			{
			t[0][ll]       = 1.0; // this has to be strictly positive !!!
			lam[0][ll]     = 1.0; // this has to be strictly positive !!!
			t[0][pnb+ll]   = 1.0; // this has to be strictly positive !!!
			lam[0][pnb+ll] = 1.0; // this has to be strictly positive !!!
			}
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<nb; ll++)
				{
				t[jj][ll]     = - db[jj][ll]     + ux[jj][ll];
				t[jj][pnb+ll] = - db[jj][pnb+ll] - ux[jj][ll];
				if(t[jj][ll] < thr0)
					{
					if(t[jj][pnb+ll] < thr0)
						{
						ux[jj][ll] = ( - db[jj][pnb+ll] + db[jj][ll])*0.5;
						t[jj][ll]     = - db[jj][ll]     + ux[jj][ll];
						t[jj][pnb+ll] = - db[jj][pnb+ll] - ux[jj][ll];
						}
					else
						{
						t[jj][ll] = thr0;
						ux[jj][ll] = db[jj][ll] + thr0;
						}
					}
				else if(t[jj][pnb+ll] < thr0)
					{
					t[jj][pnb+ll] = thr0;
					ux[jj][ll] = - db[jj][pnb+ll] - thr0;
					}
				lam[jj][ll]     = mu0/t[jj][ll];
				lam[jj][pnb+ll] = mu0/t[jj][pnb+ll];
				}
			}
		for(ll=0; ll<nbu; ll++) // this has to be strictly positive !!!
			{
			t[N][ll]       = 1.0;
			lam[N][ll]     = 1.0;
			t[N][pnb+ll]   = 1.0;
			lam[N][pnb+ll] = 1.0;
			}
		for(ll=nu; ll<nb; ll++)
			{
			t[N][ll]     = - db[N][ll]     + ux[N][ll];
			t[N][pnb+ll] = - db[N][pnb+ll] - ux[N][ll];
			if(t[N][ll] < thr0)
				{
				if(t[N][pnb+ll] < thr0)
					{
					ux[N][ll] = ( - db[N][pnb+ll] + db[N][ll])*0.5;
					t[N][ll]     = - db[N][ll]     + ux[N][ll];
					t[N][pnb+ll] = - db[N][pnb+ll] - ux[N][ll];
					}
				else
					{
					t[N][ll] = thr0;
					ux[N][ll] = db[N][ll] + thr0;
					}
				}
			else if(t[N][pnb+ll] < thr0)
				{
				t[N][pnb+ll] = thr0;
				ux[N][ll] = - db[N][pnb+ll] - thr0;
				}
			lam[N][ll] = mu0/t[N][ll];
			lam[N][pnb+ll] = mu0/t[N][pnb+ll];
			}
		}
	else // cold start
		{
		for(ll=0; ll<nbu; ll++)
			{
			ux[0][ll] = 0.0;
			//ux[0][ll] = 0.5*( - db[0][pnb+ll] + db[0][ll] );
//			t[0][ll] = 1.0;
//			t[0][pnb+ll] = 1.0;
			t[0][ll]     = - db[0][ll]     + ux[0][ll];
			t[0][pnb+ll] = - db[0][pnb+ll] - ux[0][ll];
			if(t[0][ll] < thr0)
				{
				if(t[0][pnb+ll] < thr0)
					{
					ux[0][ll] = ( - db[0][pnb+ll] + db[0][ll])*0.5;
					t[0][ll]     = - db[0][ll]     + ux[0][ll];
					t[0][pnb+ll] = - db[0][pnb+ll] - ux[0][ll];
					}
				else
					{
					t[0][ll] = thr0;
					ux[0][ll] = db[0][ll] + thr0;
					}
				}
			else if(t[0][pnb+ll] < thr0)
				{
				t[0][pnb+ll] = thr0;
				ux[0][ll] = - db[0][pnb+ll] - thr0;
				}
			lam[0][ll] = mu0/t[0][ll];
			lam[0][pnb+ll] = mu0/t[0][pnb+ll];
			}
		for(ii=ll; ii<nu; ii++)
			ux[0][ii] = 0.0; // initialize remaining components of u to zero
		for(; ll<nb; ll++)
			{
			t[0][ll]       = 1.0; // this has to be strictly positive !!!
			lam[0][ll]     = 1.0; // this has to be strictly positive !!!
			t[0][pnb+ll]   = 1.0; // this has to be strictly positive !!!
			lam[0][pnb+ll] = 1.0; // this has to be strictly positive !!!
			}
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<nb; ll++)
				{
				ux[jj][ll] = 0.0;
				//ux[jj][ll] = 0.5*( - db[jj][pnb+ll] + db[jj][ll] );
//				t[jj][ll] = 1.0;
//				t[jj][pnb+ll] = 1.0;
				t[jj][ll]     = - db[jj][ll]     + ux[jj][ll];
				t[jj][pnb+ll] = - db[jj][pnb+ll] - ux[jj][ll];
				if(t[jj][ll] < thr0)
					{
					if(t[jj][pnb+ll] < thr0)
						{
						ux[jj][ll] = ( - db[jj][pnb+ll] + db[jj][ll])*0.5;
						t[jj][ll]     = - db[jj][ll]     + ux[jj][ll];
						t[jj][pnb+ll] = - db[jj][pnb+ll] - ux[jj][ll];
						}
					else
						{
						t[jj][ll] = thr0;
						ux[jj][ll] = db[jj][ll] + thr0;
						}
					}
				else if(t[jj][pnb+ll] < thr0)
					{
					t[jj][pnb+ll] = thr0;
					ux[jj][ll] = - db[jj][pnb+ll] - thr0;
					}
				lam[jj][ll] = mu0/t[jj][ll];
				lam[jj][pnb+ll] = mu0/t[jj][pnb+ll];
				}
			for(ii=ll; ii<nx+nu; ii++)
				ux[jj][ii] = 0.0; // initialize remaining components of u and x to zero
			}
		for(ll=0; ll<nbu; ll++)
			{
			t[N][ll]       = 1.0; // this has to be strictly positive !!!
			lam[N][ll]     = 1.0; // this has to be strictly positive !!!
			t[N][pnb+ll]   = 1.0; // this has to be strictly positive !!!
			lam[N][pnb+ll] = 1.0; // this has to be strictly positive !!!
			}
		for(ll=nu; ll<nb; ll++)
			{
			ux[N][ll] = 0.0;
			//ux[N][ll] = 0.5*( - db[N][pnb+ll] + db[N][ll] );
//			t[N][ll] = 1.0;
//			t[N][pnb+ll] = 1.0;
			t[N][ll]     = - db[N][ll]     + ux[N][ll];
			t[N][pnb+ll] = - db[N][pnb+ll] - ux[N][ll];
			if(t[N][ll] < thr0)
				{
				if(t[N][pnb+ll] < thr0)
					{
					ux[N][ll] = ( - db[N][pnb+ll] + db[N][ll])*0.5;
					t[N][ll]     = - db[N][ll]     + ux[N][ll];
					t[N][pnb+ll] = - db[N][pnb+ll] - ux[N][ll];
					}
				else
					{
					t[N][ll] = thr0;
					ux[N][ll] = db[N][ll] + thr0;
					}
				}
			else if(t[N][pnb+ll] < thr0)
				{
				t[N][pnb+ll] = thr0;
				ux[N][ll] = - db[N][pnb+ll] - thr0;
				}
			lam[N][ll] = mu0/t[N][ll];
			lam[N][pnb+ll] = mu0/t[N][pnb+ll];
			}
		for(ii=ll; ii<nx+nu; ii++)
			ux[N][ii] = 0.0; // initialize remaining components of x to zero

		}

	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx; ll++)
			pi[jj][ll] = 0.0; // initialize multipliers to zero

	// TODO find a better way to initialize general constraints
	if(ng>0)
		{
		for(jj=0; jj<N; jj++)
			{

			ptr_t   = t[jj];
			ptr_lam = lam[jj];
			ptr_db  = db[jj];

			dgemv_t_lib(nx+nu, ng, pDCt[jj], cng, ux[jj], 0, ptr_t+2*pnb, ptr_t+2*pnb);

			for(ll=2*pnb; ll<2*pnb+ng; ll++)
				{
				ptr_t[ll+png] = - ptr_t[ll];
				ptr_t[ll]     += - ptr_db[ll];
				ptr_t[ll+png] += - ptr_db[ll+png];
				ptr_t[ll]     = fmax( thr0, ptr_t[ll] );
				ptr_t[png+ll] = fmax( thr0, ptr_t[png+ll] );
				ptr_lam[ll]     = mu0/ptr_t[ll];
				ptr_lam[png+ll] = mu0/ptr_t[png+ll];
				}
			}
		}
	if(ngN>0)
		{
		ptr_t   = t[N];
		ptr_lam = lam[N];
		ptr_db  = db[N];

		dgemv_t_lib(nx+nu, ngN, pDCt[N], cngN, ux[N], 0, ptr_t+2*pnb, ptr_t+2*pnb);

		for(ll=2*pnb; ll<2*pnb+ngN; ll++)
			{
			ptr_t[ll+pngN] = - ptr_t[ll];
			ptr_t[ll]      += - ptr_db[ll];
			ptr_t[ll+pngN] += - ptr_db[ll+pngN];
			ptr_t[ll]      = fmax( thr0, ptr_t[ll] );
			ptr_t[pngN+ll] = fmax( thr0, ptr_t[pngN+ll] );
			ptr_lam[ll]      = mu0/ptr_t[ll];
			ptr_lam[pngN+ll] = mu0/ptr_t[pngN+ll];
			}
		}
#if 0
d_print_mat(1, nx+nu, ux[0], 1);
d_print_mat(1, nx+nu, ux[1], 1);
d_print_mat(1, nx+nu, ux[N], 1);
d_print_mat(1, 2*pnb+2*png, t[0], 1);
d_print_mat(1, 2*pnb+2*png, t[1], 1);
d_print_mat(1, 2*pnb+2*png, t[N], 1);
d_print_mat(1, 2*pnb+2*png, lam[0], 1);
d_print_mat(1, 2*pnb+2*png, lam[1], 1);
d_print_mat(1, 2*pnb+2*png, lam[N], 1);
exit(0);
#endif

	}



void d_init_var_soft_mpc(int N, int nx, int nu, int nh, int ns, double **ux, double **pi, double **db, double **t, double **lam, double mu0, int warm_start)
	{

	int nb = nh + ns;
	
	const int nbu = nu<nb ? nu : nb ;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	int jj, ll, ii;
	
	double thr0 = 0.1; // minimum distance from a constraint

	// warm start: user-provided guess as starting point
	if(warm_start==1)
		{
		// first stage
		for(ll=0; ll<2*nbu; ll+=2)
			{
			t[0][ll+0] =   ux[0][ll/2] - db[0][ll+0];
			t[0][ll+1] = - db[0][ll+1] - ux[0][ll/2];
			if(t[0][ll+0] < thr0)
				{
				if(t[0][ll+1] < thr0)
					{
					ux[0][ll/2] = ( - db[0][ll+1] + db[0][ll+0])*0.5;
					t[0][ll+0] =   ux[0][ll/2] - db[0][ll+0];
					t[0][ll+1] = - db[0][ll+1] - ux[0][ll/2];
					}
				else
					{
					t[0][ll+0] = thr0;
					ux[0][ll/2] = db[0][ll+0] + thr0;
					}
				}
			else if(t[0][ll+1] < thr0)
				{
				t[0][ll+1] = thr0;
				ux[0][ll/2] = - db[0][ll+1] - thr0;
				}

			lam[0][ll+0] = mu0/t[0][ll+0];
			lam[0][ll+1] = mu0/t[0][ll+1];
			}
		for(; ll<2*nb; ll++)
			{
			t[0][ll] = 1.0; // this has to be strictly positive !!!
			lam[0][ll] = 1.0;
			}
		// middle stages
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
				t[jj][ll+0] =   ux[jj][ll/2] - db[jj][ll+0];
				t[jj][ll+1] = - db[jj][ll+1] - ux[jj][ll/2];
				if(t[jj][ll+0] < thr0)
					{
					if(t[jj][ll+1] < thr0)
						{
						ux[jj][ll/2] = ( - db[jj][ll+1] + db[jj][ll+0])*0.5;
						t[jj][ll+0] =   ux[jj][ll/2] - db[jj][ll+0];
						t[jj][ll+1] = - db[jj][ll+1] - ux[jj][ll/2];
						}
					else
						{
						t[jj][ll+0] = thr0;
						ux[jj][ll/2] = db[jj][ll+0] + thr0;
						}
					}
				else if(t[jj][ll+1] < thr0)
					{
					t[jj][ll+1] = thr0;
					ux[jj][ll/2] = - db[jj][ll+1] - thr0;
					}
				lam[jj][ll+0] = mu0/t[jj][ll+0];
				lam[jj][ll+1] = mu0/t[jj][ll+1];
				}
			}
		// last stage
		for(ll=0; ll<2*nbu; ll++) // this has to be strictly positive !!!
			{
			t[N][ll] = 1.0;
			lam[N][ll] = 1.0;
			}
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			t[N][ll+0] =   ux[N][ll/2] - db[N][ll+0];
			t[N][ll+1] = - db[N][ll+1] - ux[N][ll/2];
			if(t[N][ll+0] < thr0)
				{
				if(t[N][ll+1] < thr0)
					{
					ux[N][ll/2] = ( - db[N][ll+1] + db[N][ll+0])*0.5;
					t[N][ll+0] =   ux[N][ll/2] - db[N][ll+0];
					t[N][ll+1] = - db[N][ll+1] - ux[N][ll/2];
					}
				else
					{
					t[N][ll+0] = thr0;
					ux[N][ll/2] = db[N][ll+0] + thr0;
					}
				}
			else if(t[N][ll+1] < thr0)
				{
				t[N][ll+1] = thr0;
				ux[N][ll/2] = - db[N][ll+1] - thr0;
				}
			lam[N][ll+0] = mu0/t[N][ll+0];
			lam[N][ll+1] = mu0/t[N][ll+1];
			}

		}
	else // cold start : zero as starting point
		{
		// first stage
		for(ll=0; ll<2*nbu; ll+=2)
			{
			ux[0][ll/2] = 0.0;
//			t[0][ll+0] = 1.0;
//			t[0][ll+1] = 1.0;
			t[0][ll+0] =   ux[0][ll/2] - db[0][ll+0];
			t[0][ll+1] = - db[0][ll+1] - ux[0][ll/2];
			if(t[0][ll+0] < thr0)
				{
				if(t[0][ll+1] < thr0)
					{
					ux[0][ll/2] = ( - db[0][ll+1] + db[0][ll+0])*0.5;
					t[0][ll+0] =   ux[0][ll/2] - db[0][ll+0];
					t[0][ll+1] = - db[0][ll+1] - ux[0][ll/2];
					}
				else
					{
					t[0][ll+0] = thr0;
					ux[0][ll/2] = db[0][ll+0] + thr0;
					}
				}
			else if(t[0][ll+1] < thr0)
				{
				t[0][ll+1] = thr0;
				ux[0][ll/2] = - db[0][ll+1] - thr0;
				}
			lam[0][ll+0] = mu0/t[0][ll+0];
			lam[0][ll+1] = mu0/t[0][ll+1];
			}
		for(ii=ll/2; ii<nu; ii++)
			ux[0][ii] = 0.0; // initialize remaining components of u to zero
		for(; ll<2*nb; ll++)
			{
			t[0][ll] = 1.0; // this has to be strictly positive !!!
			lam[0][ll] = 1.0;
			}
		// middle stages
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
				ux[jj][ll/2] = 0.0;
//				t[jj][ll+0] = 1.0;
//				t[jj][ll+1] = 1.0;
				t[jj][ll+0] =   ux[jj][ll/2] - db[jj][ll+0];
				t[jj][ll+1] = - db[jj][ll+1] - ux[jj][ll/2];
				if(t[jj][ll+0] < thr0)
					{
					if(t[jj][ll+1] < thr0)
						{
						ux[jj][ll/2] = ( - db[jj][ll+1] + db[jj][ll+0])*0.5;
						t[jj][ll+0] =   ux[jj][ll/2] - db[jj][ll+0];
						t[jj][ll+1] = - db[jj][ll+1] - ux[jj][ll/2];
						}
					else
						{
						t[jj][ll+0] = thr0;
						ux[jj][ll/2] = db[jj][ll+0] + thr0;
						}
					}
				else if(t[jj][ll+1] < thr0)
					{
					t[jj][ll+1] = thr0;
					ux[jj][ll/2] = - db[jj][ll+1] - thr0;
					}
				lam[jj][ll+0] = mu0/t[jj][ll+0];
				lam[jj][ll+1] = mu0/t[jj][ll+1];
				}
			for(ii=ll/2; ii<nx+nu; ii++)
				ux[jj][ii] = 0.0; // initialize remaining components of u and x to zero
			}
		// last stage
		for(ll=0; ll<2*nbu; ll++)
			{
			t[N][ll] = 1.0; // this has to be strictly positive !!!
			lam[N][ll] = 1.0;
			}
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			ux[N][ll/2] = 0.0;
//			t[N][ll+0] = 1.0;
//			t[N][ll+1] = 1.0;
			t[N][ll+0] =   ux[N][ll/2] - db[N][ll+0];
			t[N][ll+1] = - db[N][ll+1] - ux[N][ll/2];
			if(t[N][ll+0] < thr0)
				{
				if(t[N][ll+1] < thr0)
					{
					ux[N][ll/2] = ( - db[N][ll+1] + db[N][ll+0])*0.5;
					t[N][ll+0] =   ux[N][ll/2] - db[N][ll+0];
					t[N][ll+1] = - db[N][ll+1] - ux[N][ll/2];
					}
				else
					{
					t[N][ll+0] = thr0;
					ux[N][ll/2] = db[N][ll+0] + thr0;
					}
				}
			else if(t[N][ll+1] < thr0)
				{
				t[N][ll+1] = thr0;
				ux[N][ll/2] = - db[N][ll+1] - thr0;
				}
			lam[N][ll+0] = mu0/t[N][ll+0];
			lam[N][ll+1] = mu0/t[N][ll+1];
			}
		for(ii=ll/2; ii<nx+nu; ii++)
			ux[N][ii] = 0.0; // initialize remaining components of x to zero

		// inizialize t_theta and lam_theta (cold start only for the moment)
		for(jj=0; jj<=N; jj++)
			for(ll=2*nh; ll<2*nb; ll++)
				{
				t[jj][pnb+ll] = 1.0;
				lam[jj][pnb+ll] = mu0; // /t[jj][pnb+ll]; // TODO restore division if needed
				}

		// initialize pi
		for(jj=0; jj<=N; jj++)
			for(ll=0; ll<nx; ll++)
				pi[jj][ll] = 0.0; // initialize multipliers to zero

		}
	
	}



void d_init_var_diag_mpc(int N, int *nx, int *nu, int *nb, int **idxb, double **ux, double **pi, double **db, double **t, double **lam, double mu0, int warm_start)
	{

	// it must be nb[0] <= nu !!!!!!!!!!!!!!!!!!!!!!!!!!

	// constants
	const int bs  = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	int jj, ll, ii;

	int pnb;
	
	double
		*ptr_t, *ptr_lam, *ptr_db;

	double thr0 = 0.1; // minimum vale of t (minimum distance from a constraint)

	// cold start
	if(warm_start==0)
		{
		for(ll=0; ll<nu[jj]; ll++)
			{
			ux[jj][ll] = 0.0;
			}
		for(jj=1; jj<=N; jj++)
			{
			for(ll=0; ll<nu[jj]+nx[jj]; ll++)
				{
				ux[jj][ll] = 0.0;
				}
			}
		}

	// check bounds & initialize multipliers
	for(jj=0; jj<=N; jj++)
		{
		pnb  = bs*((nb[jj]+bs-1)/bs); // simd aligned number of box constraints
		for(ll=0; ll<nb[jj]; ll++)
			{
			t[jj][ll]     = - db[jj][ll]     + ux[jj][idxb[jj][ll]];
			t[jj][pnb+ll] = - db[jj][pnb+ll] - ux[jj][idxb[jj][ll]];
			if(t[jj][ll] < thr0)
				{
				if(t[jj][pnb+ll] < thr0)
					{
					ux[jj][idxb[jj][ll]] = ( - db[jj][pnb+ll] + db[jj][ll])*0.5;
					t[jj][ll]     = - db[jj][ll]     + ux[jj][idxb[jj][ll]];
					t[jj][pnb+ll] = - db[jj][pnb+ll] - ux[jj][idxb[jj][ll]];
					}
				else
					{
					t[jj][ll] = thr0;
					ux[jj][idxb[jj][ll]] = db[jj][ll] + thr0;
					}
				}
			else if(t[jj][pnb+ll] < thr0)
				{
				t[jj][pnb+ll] = thr0;
				ux[jj][idxb[jj][ll]] = - db[jj][pnb+ll] - thr0;
				}
			lam[jj][ll]     = mu0/t[jj][ll];
			lam[jj][pnb+ll] = mu0/t[jj][pnb+ll];
			}
		}

	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx[jj]; ll++)
			pi[jj][ll] = 0.0; // initialize multipliers to zero

	}



void d_update_hessian_hard_mpc_tv(int N, int *nx, int *nu, int *nb, int *ng, double sigma_mu, double **t, double **tinv, double **lam, double **lamt, double **dlam, double **Qx, double **qx, double **qx2, double **bd, double **bl, double **pd, double **pl, double **db)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	__m256d
		v_ones, v_sigma_mu, v_mask, v_left,
		v_tmp, v_lam, v_lamt, v_dlam, v_db,
		v_tmp0, v_tmp1,
		v_lam0, v_lam1,
		v_lamt0, v_lamt1,
		v_dlam0, v_dlam1,
		v_Qx0, v_Qx1,
		v_qx0, v_qx1,
		v_bd0, v_bd2,
		v_db0, v_db2;
	
	__m256i
		i_mask;

	double 
		*ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Qx, *ptr_qx, *ptr_qx2,
		*ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );

	int ii, jj, bs0;

	double ii_left;

	int nb0, pnb, ng0, png;
	
	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	for(jj=0; jj<=N; jj++)
		{
		
		nb0 = nb[jj];
		pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints
		ng0 = ng[jj];
		png  = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints

		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_tinv  = tinv[jj];
		ptr_db    = db[jj];
		ptr_bd    = bd[jj];
		ptr_bl    = bl[jj];
		ptr_pd    = pd[jj];
		ptr_pl    = pl[jj];
		ptr_Qx    = Qx[jj];
		ptr_qx    = qx[jj];
		ptr_qx2   = qx2[jj];

		// box constraints
		ii = 0;
		for(; ii<nb0-3; ii+=4)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[ii+pnb] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[ii+pnb] );
			v_qx0   = _mm256_load_pd( &ptr_db[ii] );
			v_qx1   = _mm256_load_pd( &ptr_db[pnb+ii] );
			_mm256_store_pd( &ptr_tinv[ii+0], v_tmp0 ); // store tinv
			_mm256_store_pd( &ptr_tinv[ii+pnb], v_tmp1 ); // store tinv
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_store_pd( &ptr_lamt[ii+0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[ii+pnb], v_lamt1 );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			_mm256_store_pd( &ptr_dlam[ii+0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[ii+pnb], v_dlam1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

			v_lamt0 = _mm256_add_pd( v_lamt0, v_lamt1 );
			v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
			v_tmp0  = _mm256_load_pd( &ptr_bd[ii] );
			v_tmp1  = _mm256_load_pd( &ptr_bl[ii] );
			v_tmp0  = _mm256_add_pd( v_lamt0, v_tmp0 );
			v_tmp1  = _mm256_add_pd( v_tmp1, v_qx0 );
			_mm256_store_pd( &ptr_pd[ii], v_tmp0 );
			_mm256_store_pd( &ptr_pl[ii], v_tmp1 );

			}
		if(ii<nb0)
			{

			ii_left = nb0-ii;
			v_left= _mm256_broadcast_sd( &ii_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[ii+pnb] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_maskstore_pd( &ptr_tinv[ii+0], i_mask, v_tmp0 ); // store tinv
			_mm256_maskstore_pd( &ptr_tinv[ii+pnb], i_mask, v_tmp1 ); // store tinv
			v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[ii+pnb] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_maskstore_pd( &ptr_lamt[ii+0], i_mask, v_lamt0 );
			_mm256_maskstore_pd( &ptr_lamt[ii+pnb], i_mask, v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_maskstore_pd( &ptr_dlam[ii+0], i_mask, v_dlam0 );
			_mm256_maskstore_pd( &ptr_dlam[ii+pnb], i_mask, v_dlam1 );

			v_Qx0   = v_lamt0;
			v_Qx1   = v_lamt1;
			v_qx0   = _mm256_load_pd( &ptr_db[ii] );
			v_qx1   = _mm256_load_pd( &ptr_db[pnb+ii] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

			v_Qx0   = _mm256_add_pd( v_Qx0, v_Qx1 );
			v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
			v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
			v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
			v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
			v_tmp1  = _mm256_add_pd( v_tmp1, v_qx0 );
			_mm256_maskstore_pd( &pd[jj][ii], i_mask, v_tmp0 );
			_mm256_maskstore_pd( &pl[jj][ii], i_mask, v_tmp1 );

			}
		//for( ; ii<nu[jj]+nx[jj]; ii++)
		//	{
		//	ptr_pd[ii] = ptr_bd[ii];
		//	ptr_pl[ii] = ptr_bl[ii];
		//	}
	
		// general constraints
		for(ii=2*pnb; ii<2*pnb+ng0-3; ii+=4)
			{
			
			v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[ii+png] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_store_pd( &ptr_tinv[ii+0], v_tmp0 );
			_mm256_store_pd( &ptr_tinv[ii+png], v_tmp1 );
			v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[ii+png] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_store_pd( &ptr_lamt[ii+0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[ii+png], v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[ii+0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[ii+png], v_dlam1 );

			v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
			v_Qx0   = _mm256_sqrt_pd( v_Qx0 );
			_mm256_store_pd( &ptr_Qx[ii+0], v_Qx0 );
			v_qx0   = _mm256_load_pd( &ptr_db[ii+0] );
			v_qx1   = _mm256_load_pd( &ptr_db[ii+png] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );
			v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
			_mm256_store_pd( &ptr_qx[ii+0], v_qx0 );
			v_qx0   = _mm256_div_pd( v_qx0, v_Qx0 );
			_mm256_store_pd( &ptr_qx2[ii+0], v_qx0 );

			}
		if(ii<2*pnb+ng0)
			{

			ii_left = 2*pnb + ng0 - ii;
			v_left  = _mm256_broadcast_sd( &ii_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			i_mask  = _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[ii+png] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_maskstore_pd( &ptr_tinv[ii+0], i_mask, v_tmp0 );
			_mm256_maskstore_pd( &ptr_tinv[ii+png], i_mask, v_tmp1 );
			v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[ii+png] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_maskstore_pd( &ptr_lamt[ii+0], i_mask, v_lamt0 );
			_mm256_maskstore_pd( &ptr_lamt[ii+png], i_mask, v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_maskstore_pd( &ptr_dlam[ii+0], i_mask, v_dlam0 );
			_mm256_maskstore_pd( &ptr_dlam[ii+png], i_mask, v_dlam1 );

			v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
			v_Qx0   = _mm256_sqrt_pd( v_Qx0 );
			_mm256_maskstore_pd( &ptr_Qx[ii+0], i_mask, v_Qx0 );
			v_qx0   = _mm256_load_pd( &ptr_db[ii] );
			v_qx1   = _mm256_load_pd( &ptr_db[png+ii] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );
			v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
			_mm256_maskstore_pd( &ptr_qx[ii+0], i_mask, v_qx0 );
			v_qx0   = _mm256_div_pd( v_qx0, v_Qx0 );
			_mm256_maskstore_pd( &ptr_qx2[ii+0], i_mask, v_qx0 );

			}

		}

	}



void d_update_hessian_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **Qx, double **qx, double **bd, double **bl, double **pd, double **pl, double **db)
	{

	const int nbu = nu<nb ? nu : nb ;

	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	//const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!
	const int pnb  = bs*((nb+bs-1)/bs); // simd aligned number of box constraints
	const int png  = bs*((ng+bs-1)/bs); // simd aligned number of general constraints
	const int pngN = bs*((ngN+bs-1)/bs); // simd aligned number of general constraints
	//printf("\n%d %d\n", nb, ng);
	
	__m256d
		v_ones, v_sigma_mu, v_mask, v_left,
		v_tmp, v_lam, v_lamt, v_dlam, v_db,
		v_tmp0, v_tmp1, v_tmp2, v_tmp3, 
		v_lam0, v_lam1, v_lam2, v_lam3,
		v_lamt0, v_lamt1, v_lamt2, v_lamt3,
		v_dlam0, v_dlam1, v_dlam2, v_dlam3,
		v_Qx0, v_Qx1, v_Qx2, v_Qx3, 
		v_qx0, v_qx1, v_qx2, v_qx3,
		v_bd0, v_bd2,
		v_db0, v_db2;
			
	__m128d
		u_tmp0, u_tmp1,
		u_Qx0, u_Qx1, u_qx0, u_qx1,
		u_lamt0, u_bd, u_bl, u_lam0, u_dlam0, u_db0,
		u_lamt1, u_lam1, u_dlam1, u_db1;
	
	__m256i
		i_mask;

	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );
	
	double 
		*ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Qx, *ptr_qx,
		*ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_t_inv;
	
	int ii, jj, ll, bs0;
	double ii_left, ii_start;
	
	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	// first stage
	jj = 0;
	
	ptr_t     = t[jj];
	ptr_lam   = lam[jj];
	ptr_lamt  = lamt[jj];
	ptr_dlam  = dlam[jj];
	ptr_t_inv = t_inv[jj];
	ptr_db    = db[jj];
	ptr_bd    = bd[jj];
	ptr_bl    = bl[jj];
	ptr_pd    = pd[jj];
	ptr_pl    = pl[jj];

	ii = 0;
	for(; ii<nbu-3; ii+=4)
		{
		
		v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[ii+pnb] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[ii+pnb] );
		v_qx0   = _mm256_load_pd( &ptr_db[ii] );
		v_qx1   = _mm256_load_pd( &ptr_db[pnb+ii] );
		_mm256_store_pd( &ptr_t_inv[ii+0], v_tmp0 ); // store t_inv
		_mm256_store_pd( &ptr_t_inv[ii+pnb], v_tmp1 ); // store t_inv
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		_mm256_store_pd( &ptr_lamt[ii+0], v_lamt0 );
		_mm256_store_pd( &ptr_lamt[ii+pnb], v_lamt1 );
		v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		_mm256_store_pd( &ptr_dlam[ii+0], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[ii+pnb], v_dlam1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

		v_lamt0 = _mm256_add_pd( v_lamt0, v_lamt1 );
		v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
		v_tmp0  = _mm256_load_pd( &ptr_bd[ii] );
		v_tmp1  = _mm256_load_pd( &ptr_bl[ii] );
		v_tmp0  = _mm256_add_pd( v_lamt0, v_tmp0 );
		v_tmp1  = _mm256_add_pd( v_tmp1, v_qx0 );
		_mm256_store_pd( &ptr_pd[ii], v_tmp0 );
		_mm256_store_pd( &ptr_pl[ii], v_tmp1 );

		}
	if(ii<nbu)
		{

		ii_left = nbu-ii;
		v_left= _mm256_broadcast_sd( &ii_left );
		v_mask= _mm256_loadu_pd( d_mask );
		i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

		v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[ii+pnb] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		_mm256_maskstore_pd( &ptr_t_inv[ii+0], i_mask, v_tmp0 ); // store t_inv
		_mm256_maskstore_pd( &ptr_t_inv[ii+pnb], i_mask, v_tmp1 ); // store t_inv
		v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[ii+pnb] );
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		_mm256_maskstore_pd( &ptr_lamt[ii+0], i_mask, v_lamt0 );
		_mm256_maskstore_pd( &ptr_lamt[ii+pnb], i_mask, v_lamt1 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		_mm256_maskstore_pd( &ptr_dlam[ii+0], i_mask, v_dlam0 );
		_mm256_maskstore_pd( &ptr_dlam[ii+pnb], i_mask, v_dlam1 );

		v_Qx0   = v_lamt0;
		v_Qx1   = v_lamt1;
		v_qx0   = _mm256_load_pd( &ptr_db[ii] );
		v_qx1   = _mm256_load_pd( &ptr_db[pnb+ii] );
		v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

		v_Qx0   = _mm256_add_pd( v_Qx0, v_Qx1 );
		v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
		v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
		v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
		v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
		v_tmp1  = _mm256_add_pd( v_tmp1, v_qx0 );
		_mm256_maskstore_pd( &pd[jj][ii], i_mask, v_tmp0 );
		_mm256_maskstore_pd( &pl[jj][ii], i_mask, v_tmp1 );

		ii += ii_left;

		}
	for( ; ii<nu; ii++)
		{
		ptr_pd[ii] = ptr_bd[ii];
		ptr_pl[ii] = ptr_bl[ii];
		}
	
	// middle stages
	for(jj=1; jj<N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_t_inv = t_inv[jj];
		ptr_db    = db[jj];
		ptr_bd    = bd[jj];
		ptr_bl    = bl[jj];
		ptr_pd    = pd[jj];
		ptr_pl    = pl[jj];

		ii = 0;
		for(; ii<nb-3; ii+=4)
			{
		
			v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[ii+pnb] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[ii+pnb] );
			v_qx0   = _mm256_load_pd( &ptr_db[ii] );
			v_qx1   = _mm256_load_pd( &ptr_db[pnb+ii] );
			_mm256_store_pd( &ptr_t_inv[ii+0], v_tmp0 ); // store t_inv
			_mm256_store_pd( &ptr_t_inv[ii+pnb], v_tmp1 ); // store t_inv
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_store_pd( &ptr_lamt[ii+0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[ii+pnb], v_lamt1 );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			_mm256_store_pd( &ptr_dlam[ii+0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[ii+pnb], v_dlam1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

			v_lamt0 = _mm256_add_pd( v_lamt0, v_lamt1 );
			v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
			v_tmp0  = _mm256_load_pd( &ptr_bd[ii] );
			v_tmp1  = _mm256_load_pd( &ptr_bl[ii] );
			v_tmp0  = _mm256_add_pd( v_lamt0, v_tmp0 );
			v_tmp1  = _mm256_add_pd( v_tmp1, v_qx0 );
			_mm256_store_pd( &ptr_pd[ii], v_tmp0 );
			_mm256_store_pd( &ptr_pl[ii], v_tmp1 );

			}
		if(ii<nb)
			{

			ii_left = nb-ii;
			v_left= _mm256_broadcast_sd( &ii_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[ii+pnb] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_maskstore_pd( &ptr_t_inv[ii+0], i_mask, v_tmp0 ); // store t_inv
			_mm256_maskstore_pd( &ptr_t_inv[ii+pnb], i_mask, v_tmp1 ); // store t_inv
			v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[ii+pnb] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_maskstore_pd( &ptr_lamt[ii+0], i_mask, v_lamt0 );
			_mm256_maskstore_pd( &ptr_lamt[ii+pnb], i_mask, v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_maskstore_pd( &ptr_dlam[ii+0], i_mask, v_dlam0 );
			_mm256_maskstore_pd( &ptr_dlam[ii+pnb], i_mask, v_dlam1 );

			v_Qx0   = v_lamt0;
			v_Qx1   = v_lamt1;
			v_qx0   = _mm256_load_pd( &ptr_db[ii] );
			v_qx1   = _mm256_load_pd( &ptr_db[pnb+ii] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

			v_Qx0   = _mm256_add_pd( v_Qx0, v_Qx1 );
			v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
			v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
			v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
			v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
			v_tmp1  = _mm256_add_pd( v_tmp1, v_qx0 );
			_mm256_maskstore_pd( &pd[jj][ii], i_mask, v_tmp0 );
			_mm256_maskstore_pd( &pl[jj][ii], i_mask, v_tmp1 );

			ii += ii_left;

			}
		for( ; ii<nu+nx; ii++)
			{
			ptr_pd[ii] = ptr_bd[ii];
			ptr_pl[ii] = ptr_bl[ii];
			}
			
		}

	// last stage
	jj = N;

	ptr_t     = t[jj];
	ptr_lam   = lam[jj];
	ptr_lamt  = lamt[jj];
	ptr_dlam  = dlam[jj];
	ptr_t_inv = t_inv[jj];
	ptr_db    = db[jj];
	ptr_bd    = bd[jj];
	ptr_bl    = bl[jj];
	ptr_pd    = pd[jj];
	ptr_pl    = pl[jj];

	ii = (nu/4)*4;

	ii_start = nu%4; // number of not store at the beginning
	if(nb>nu && ii_start!=0)
		{

		v_mask  = _mm256_loadu_pd( d_mask );
		v_tmp0  = _mm256_broadcast_sd( &ii_start );
		v_tmp0  = _mm256_sub_pd( v_tmp0, v_mask );
		ii_left = nb - ii;
		v_left  = _mm256_broadcast_sd( &ii_left );
		v_left  = _mm256_sub_pd( v_mask, v_left );
		v_left  = _mm256_and_pd( v_left, v_tmp0 );
		i_mask  = _mm256_castpd_si256( v_left );

		v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[ii+pnb] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		_mm256_maskstore_pd( &ptr_t_inv[ii+0], i_mask, v_tmp0 ); // store t_inv
		_mm256_maskstore_pd( &ptr_t_inv[ii+pnb], i_mask, v_tmp1 ); // store t_inv
		v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[ii+pnb] );
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		_mm256_maskstore_pd( &ptr_lamt[ii+0], i_mask, v_lamt0 );
		_mm256_maskstore_pd( &ptr_lamt[ii+pnb], i_mask, v_lamt1 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		_mm256_maskstore_pd( &ptr_dlam[ii+0], i_mask, v_dlam0 );
		_mm256_maskstore_pd( &ptr_dlam[ii+pnb], i_mask, v_dlam1 );

		v_Qx0   = v_lamt0;
		v_Qx1   = v_lamt1;
		v_qx0   = _mm256_load_pd( &ptr_db[ii] );
		v_qx1   = _mm256_load_pd( &ptr_db[pnb+ii] );
		v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

		v_Qx0   = _mm256_add_pd( v_Qx0, v_Qx1 );
		v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
		v_tmp0  = _mm256_load_pd( &ptr_bd[ii] );
		v_tmp1  = _mm256_load_pd( &ptr_bl[ii] );
		v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
		v_tmp1  = _mm256_add_pd( v_tmp1, v_qx0 );
		_mm256_maskstore_pd( &ptr_pd[ii], i_mask, v_tmp0 );
		_mm256_maskstore_pd( &ptr_pl[ii], i_mask, v_tmp1 );

		ii += 4;
		if(nb<ii) ii=nb;

		}

	for(; ii<nb-3; ii+=4)
		{
		
		v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[ii+pnb] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[ii+pnb] );
		v_qx0   = _mm256_load_pd( &ptr_db[ii] );
		v_qx1   = _mm256_load_pd( &ptr_db[pnb+ii] );
		_mm256_store_pd( &ptr_t_inv[ii+0], v_tmp0 ); // store t_inv
		_mm256_store_pd( &ptr_t_inv[ii+pnb], v_tmp1 ); // store t_inv
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		_mm256_store_pd( &ptr_lamt[ii+0], v_lamt0 );
		_mm256_store_pd( &ptr_lamt[ii+pnb], v_lamt1 );
		v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		_mm256_store_pd( &ptr_dlam[ii+0], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[ii+pnb], v_dlam1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

		v_lamt0 = _mm256_add_pd( v_lamt0, v_lamt1 );
		v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
		v_tmp0  = _mm256_load_pd( &ptr_bd[ii] );
		v_tmp1  = _mm256_load_pd( &ptr_bl[ii] );
		v_tmp0  = _mm256_add_pd( v_lamt0, v_tmp0 );
		v_tmp1  = _mm256_add_pd( v_tmp1, v_qx0 );
		_mm256_store_pd( &ptr_pd[ii], v_tmp0 );
		_mm256_store_pd( &ptr_pl[ii], v_tmp1 );

		}
	if(ii<nb)
		{

		ii_left = nb-ii;
		v_left= _mm256_broadcast_sd( &ii_left );
		v_mask= _mm256_loadu_pd( d_mask );
		i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

		v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[ii+pnb] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		_mm256_maskstore_pd( &ptr_t_inv[ii+0], i_mask, v_tmp0 ); // store t_inv
		_mm256_maskstore_pd( &ptr_t_inv[ii+pnb], i_mask, v_tmp1 ); // store t_inv
		v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[ii+pnb] );
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		_mm256_maskstore_pd( &ptr_lamt[ii+0], i_mask, v_lamt0 );
		_mm256_maskstore_pd( &ptr_lamt[ii+pnb], i_mask, v_lamt1 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		_mm256_maskstore_pd( &ptr_dlam[ii+0], i_mask, v_dlam0 );
		_mm256_maskstore_pd( &ptr_dlam[ii+pnb], i_mask, v_dlam1 );

		v_Qx0   = v_lamt0;
		v_Qx1   = v_lamt1;
		v_qx0   = _mm256_load_pd( &ptr_db[ii] );
		v_qx1   = _mm256_load_pd( &ptr_db[pnb+ii] );
		v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

		v_Qx0   = _mm256_add_pd( v_Qx0, v_Qx1 );
		v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
		v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
		v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
		v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
		v_tmp1  = _mm256_add_pd( v_tmp1, v_qx0 );
		_mm256_maskstore_pd( &pd[jj][ii], i_mask, v_tmp0 );
		_mm256_maskstore_pd( &pl[jj][ii], i_mask, v_tmp1 );

		ii += ii_left;

		}
	for( ; ii<nu+nx; ii++)
		{
		ptr_pd[ii] = ptr_bd[ii];
		ptr_pl[ii] = ptr_bl[ii];
		}

	// general constraints
	if(ng>0)
		{

		for(jj=0; jj<N; jj++)
			{

			ptr_t     = t[jj];
			ptr_lam   = lam[jj];
			ptr_lamt  = lamt[jj];
			ptr_dlam  = dlam[jj];
			ptr_t_inv = t_inv[jj];
			ptr_db    = db[jj];
			ptr_bd    = bd[jj];
			ptr_bl    = bl[jj];
			ptr_pd    = pd[jj];
			ptr_pl    = pl[jj];
			ptr_Qx    = Qx[jj];
			ptr_qx    = qx[jj];

			for(ii=2*pnb; ii<2*pnb+ng-3; ii+=4)
				{
				
				v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
				v_tmp1  = _mm256_load_pd( &ptr_t[ii+png] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
				_mm256_store_pd( &ptr_t_inv[ii+0], v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[ii+png], v_tmp1 );
				v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_load_pd( &ptr_lam[ii+png] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
				_mm256_store_pd( &ptr_lamt[ii+0], v_lamt0 );
				_mm256_store_pd( &ptr_lamt[ii+png], v_lamt1 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[ii+0], v_dlam0 );
				_mm256_store_pd( &ptr_dlam[ii+png], v_dlam1 );

				v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
				v_Qx0   = _mm256_sqrt_pd( v_Qx0 );
				_mm256_store_pd( &ptr_Qx[ii+0], v_Qx0 );
	 			v_qx0   = _mm256_load_pd( &ptr_db[ii+0] );
				v_qx1   = _mm256_load_pd( &ptr_db[ii+png] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
				v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );
				v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
				_mm256_store_pd( &ptr_qx[ii+0], v_qx0 );
				//_mm256_store_pd( &ptr_qx[ii+png], v_qx1 );

				}
			if(ii<2*pnb+ng)
				{

				ii_left = 2*pnb + ng - ii;
				v_left  = _mm256_broadcast_sd( &ii_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				i_mask  = _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

				v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
				v_tmp1  = _mm256_load_pd( &ptr_t[ii+png] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
				_mm256_maskstore_pd( &ptr_t_inv[ii+0], i_mask, v_tmp0 );
				_mm256_maskstore_pd( &ptr_t_inv[ii+png], i_mask, v_tmp1 );
				v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_load_pd( &ptr_lam[ii+png] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
				_mm256_maskstore_pd( &ptr_lamt[ii+0], i_mask, v_lamt0 );
				_mm256_maskstore_pd( &ptr_lamt[ii+png], i_mask, v_lamt1 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
				_mm256_maskstore_pd( &ptr_dlam[ii+0], i_mask, v_dlam0 );
				_mm256_maskstore_pd( &ptr_dlam[ii+png], i_mask, v_dlam1 );

				v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
				v_Qx0   = _mm256_sqrt_pd( v_Qx0 );
				_mm256_maskstore_pd( &ptr_Qx[ii+0], i_mask, v_Qx0 );
				v_qx0   = _mm256_load_pd( &ptr_db[ii] );
				v_qx1   = _mm256_load_pd( &ptr_db[png+ii] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
				v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );
				v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
				_mm256_maskstore_pd( &ptr_qx[ii+0], i_mask, v_qx0 );
				//_mm256_maskstore_pd( &ptr_qx[ii+png], i_mask, v_qx1 );

				ii += ii_left;

				}

			}

		}
	if(ngN>0)
		{

		ptr_t     = t[N];
		ptr_lam   = lam[N];
		ptr_lamt  = lamt[N];
		ptr_dlam  = dlam[N];
		ptr_t_inv = t_inv[N];
		ptr_db    = db[N];
		ptr_bd    = bd[N];
		ptr_bl    = bl[N];
		ptr_pd    = pd[N];
		ptr_pl    = pl[N];
		ptr_Qx    = Qx[N];
		ptr_qx    = qx[N];

		for(ii=2*pnb; ii<2*pnb+ngN-3; ii+=4)
			{
			
			v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[ii+pngN] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_store_pd( &ptr_t_inv[ii+0], v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[ii+pngN], v_tmp1 );
			v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[ii+pngN] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_store_pd( &ptr_lamt[ii+0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[ii+pngN], v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[ii+0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[ii+pngN], v_dlam1 );

			v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
			v_Qx0   = _mm256_sqrt_pd( v_Qx0 );
			_mm256_store_pd( &ptr_Qx[ii+0], v_Qx0 );
			v_qx0   = _mm256_load_pd( &ptr_db[ii+0] );
			v_qx1   = _mm256_load_pd( &ptr_db[ii+pngN] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );
			v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
			_mm256_store_pd( &ptr_qx[ii+0], v_qx0 );
			//_mm256_store_pd( &ptr_qx[ii+pngN], v_qx1 );

			}
		if(ii<2*pnb+ngN)
			{

			ii_left = 2*pnb + ngN - ii;
			v_left  = _mm256_broadcast_sd( &ii_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			i_mask  = _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_tmp0  = _mm256_load_pd( &ptr_t[ii+0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[ii+pngN] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_maskstore_pd( &ptr_t_inv[ii+0], i_mask, v_tmp0 );
			_mm256_maskstore_pd( &ptr_t_inv[ii+pngN], i_mask, v_tmp1 );
			v_lam0  = _mm256_load_pd( &ptr_lam[ii+0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[ii+pngN] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_maskstore_pd( &ptr_lamt[ii+0], i_mask, v_lamt0 );
			_mm256_maskstore_pd( &ptr_lamt[ii+pngN], i_mask, v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_maskstore_pd( &ptr_dlam[ii+0], i_mask, v_dlam0 );
			_mm256_maskstore_pd( &ptr_dlam[ii+pngN], i_mask, v_dlam1 );

			v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
			v_Qx0   = _mm256_sqrt_pd( v_Qx0 );
			_mm256_maskstore_pd( &ptr_Qx[ii+0], i_mask, v_Qx0 );
			v_qx0   = _mm256_load_pd( &ptr_db[ii] );
			v_qx1   = _mm256_load_pd( &ptr_db[pngN+ii] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );
			v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
			_mm256_maskstore_pd( &ptr_qx[ii+0], i_mask, v_qx0 );
			//_mm256_maskstore_pd( &ptr_qx[ii+pngN], i_mask, v_qx1 );

			ii += ii_left;

			}

		}

	
	return;

	}



void d_update_hessian_soft_mpc(int N, int nx, int nu, int nh, int ns, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **db, double **Z, double **z, double **Zl, double **zl)
	{

	int nb = nh + ns;

	int nbu = nu<nb ? nu : nb ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	//const int k0 = nbu;
	//const int k1 = (nu/bs)*bs;
	//const int kmax = nb;
	
	__m256d
		v_zeros, v_ones, v_sigma_mu,
		v_tmp0, v_tmp1, v_tmp2, v_tmp3, 
		v_lam0, v_lam1, v_lam2, v_lam3,
		v_lamt0, v_lamt1, v_lamt2, v_lamt3,
		v_dlam0, v_dlam1, v_dlam2, v_dlam3,
		v_Qx0, v_Qx1, v_qx0, v_qx1,
		v_Zl0, v_Zl1, v_zl0, v_zl1,
		v_bd0, v_bd2,
		v_db0, v_db2;
		
	__m128d
		u_tmp, u_lamt, u_bd, u_bl, u_lam, u_dlam, u_db,
		u_lam0, u_lam1, u_dlam0, u_dlam1, u_lamt0, u_lamt1,
		u_tmp0, u_tmp1, u_Qx, u_qx, u_Zl, u_zl;
	
	__m256i
		i_mask;
	
	v_zeros    = _mm256_setzero_pd();
	v_ones     = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );

	const long long mask2[] = { 1, 1, -1, -1 };
		
	double temp0, temp1;
	
	double *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_t_inv, 
		*ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Z, *ptr_z, *ptr_Zl, *ptr_zl;

//	static double Qx[8] = {};
//	static double qx[8] = {};
	
	int ii, jj, ll, bs0;
	
	// first stage
	jj = 0;
	
	ptr_t     = t[0];
	ptr_lam   = lam[0];
	ptr_lamt  = lamt[0];
	ptr_dlam  = dlam[0];
	ptr_t_inv = t_inv[0];
	
	ii = 0;
	// hard constraints on u only
	for(; ii<nbu-3; ii+=4)
		{

		v_tmp0  = _mm256_load_pd( &ptr_t[0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[4] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp0 ); // store t_inv
		_mm256_store_pd( &ptr_t_inv[4], v_tmp1 ); // store t_inv
		v_lam0  = _mm256_load_pd( &ptr_lam[0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[4] );
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
		_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[4], v_dlam1 );

		v_Qx0   = v_lamt0;
		v_Qx1   = v_lamt1;
		v_qx0   = _mm256_load_pd( &db[jj][2*ii+0] );
		v_qx1   = _mm256_load_pd( &db[jj][2*ii+4] );
		v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_dlam1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

		v_tmp0  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
		v_Qx1   = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
		v_Qx0   = v_tmp0;
		v_tmp1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
		v_qx1   = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
		v_qx0   = v_tmp1;
		v_Qx0   = _mm256_hadd_pd( v_Qx0, v_Qx1 );
		v_qx0   = _mm256_hsub_pd( v_qx0, v_qx1 );
		v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
		v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
		v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
		v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
		_mm256_store_pd( &pd[jj][ii], v_tmp0 );
		_mm256_store_pd( &pl[jj][ii], v_tmp1 );

		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;

		}
	if(ii<nbu)
		{
		bs0 = nbu-ii;
		ll = 0;
		
		if(bs0>=2)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

			v_Qx0   = v_lamt0;
			v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

			u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
			u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
			u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
			u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
			u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
			u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
			u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
			u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
			_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

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
			
			u_tmp0 = _mm_load_pd( &ptr_t[0] );
			u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
			_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp0, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );

			u_Qx   = u_lamt;
			u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
			u_qx   = _mm_mul_pd( u_qx, u_lamt );
			u_qx   = _mm_add_pd( u_qx, u_dlam );
			u_qx   = _mm_add_pd( u_qx, u_lam );

			u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
			u_qx   = _mm_hsub_pd( u_qx, u_qx );
			u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
			u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
			u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
			u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
			_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/
			
			ll++;

			}
		ii += ll;
		}
	for( ; ii<nu; ii++)
		{
		pd[jj][ii] = bd[jj][ii];
		pl[jj][ii] = bl[jj][ii];
		}


	// middle stages

	for(jj=1; jj<N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_t_inv = t_inv[jj];

		ptr_pd    = pd[jj];
		ptr_pl    = pl[jj];
		ptr_bd    = bd[jj];
		ptr_bl    = bl[jj];
		ptr_db    = db[jj];
		ptr_Z     = Z[jj];
		ptr_z     = z[jj];
		ptr_Zl    = Zl[jj];
		ptr_zl    = zl[jj];

		ii = 0;
		// hard constraints on u and x
		for(; ii<nh-3; ii+=4)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[4] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 ); // store t_inv
			_mm256_store_pd( &ptr_t_inv[4], v_tmp1 ); // store t_inv
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[4] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[4], v_dlam1 );

			v_Qx0  = v_lamt0;
			v_Qx1  = v_lamt1;
			v_qx0  = _mm256_load_pd( &db[jj][2*ii+0] );
			v_qx1  = _mm256_load_pd( &db[jj][2*ii+4] );
			v_qx0  = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1  = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_dlam1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_lam1 );

			v_tmp0 = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
			v_Qx1  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
			v_Qx0  = v_tmp0;
			v_tmp1 = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
			v_qx1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
			v_qx0  = v_tmp1;
			v_Qx0  = _mm256_hadd_pd( v_Qx0, v_Qx1 );
			v_qx0  = _mm256_hsub_pd( v_qx0, v_qx1 );
			v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
			v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
			v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
			_mm256_store_pd( &pd[jj][ii], v_tmp0 );
			_mm256_store_pd( &pl[jj][ii], v_tmp1 );

			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_t_inv += 8;

			ptr_db    += 8;
			ptr_Z     += 8;
			ptr_z     += 8;
			ptr_Zl    += 8;
			ptr_zl    += 8;

			}
		if(ii<nh)
			{
			// clean-up loop
			bs0 = nh-ii;
			ll = 0;
			if(bs0>=2)
				{

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;
				
				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;
				
				ll   += 2;
				bs0  -= 2;

				}
			
			if(bs0>0)
				{
				if(nh<nb) // there are soft constraints afterwards
					{

					v_tmp0  = _mm256_load_pd( &ptr_t[0] );
					u_tmp1  = _mm_load_pd( &ptr_t[pnb+2] );
					v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
					u_tmp1  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp1 );
					_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
					_mm_store_pd( &ptr_t_inv[pnb+2], u_tmp1 );
					v_lam0  = _mm256_load_pd( &ptr_lam[0] );
					u_lam1  = _mm_load_pd( &ptr_lam[pnb+2] );
					v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
					u_lamt1 = _mm_mul_pd( u_tmp1, u_lam1 );
					_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
					_mm_store_pd( &ptr_lamt[pnb+2], u_lamt1 );
					v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
					u_dlam1 = _mm_mul_pd( u_tmp1, _mm256_castpd256_pd128( v_sigma_mu ) );
					_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
					_mm_store_pd( &ptr_dlam[pnb+2], u_dlam1 );

					v_Qx0   = v_lamt0;
					v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
					v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
					v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
					v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

					u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
					u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
					u_Zl   = _mm_load_pd( &ptr_Z[2] );
					u_zl   = _mm_load_pd( &ptr_z[2] );
					u_Zl   = _mm_add_pd( u_Zl, u_Qx );
					u_Zl   = _mm_add_pd( u_Zl, u_lamt1 );
					u_Zl   = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_Zl );
					u_zl   = _mm_sub_pd( u_qx, u_zl );
					u_zl   = _mm_add_pd( u_zl, u_lam1 );
					u_zl   = _mm_add_pd( u_zl, u_dlam1 );
					_mm_store_pd( &ptr_Zl[2], u_Zl );
					_mm_store_pd( &ptr_zl[2], u_zl );
					u_tmp0 = _mm_mul_pd( u_Qx, u_Zl );
					u_tmp1 = _mm_mul_pd( u_tmp0, u_zl );
					u_qx  = _mm_sub_pd( u_qx, u_tmp1 );
					u_tmp0 = _mm_mul_pd( u_Qx, u_tmp0 );
					u_Qx  = _mm_sub_pd( u_Qx, u_tmp0 );

					u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
					u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
					u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
					u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
					u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
					u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
					_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
					_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

					ptr_t     += 4;
					ptr_lam   += 4;
					ptr_lamt  += 4;
					ptr_dlam  += 4;
					ptr_t_inv += 4;

					ptr_db    += 4;
					ptr_Z     += 4;
					ptr_z     += 4;
					ptr_Zl    += 4;
					ptr_zl    += 4;
				
					ll   += 2;
					}
				else // no soft constraints afterward
					{

					u_tmp0 = _mm_load_pd( &ptr_t[0] );
					u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
					_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
					u_lam  = _mm_load_pd( &ptr_lam[0] );
					u_lamt = _mm_mul_pd( u_tmp0, u_lam );
					_mm_store_pd( &ptr_lamt[0], u_lamt );
					u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
					_mm_store_pd( &ptr_dlam[0], u_dlam );

					u_Qx   = u_lamt;
					u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
					u_qx   = _mm_mul_pd( u_qx, u_lamt );
					u_qx   = _mm_add_pd( u_qx, u_dlam );
					u_qx   = _mm_add_pd( u_qx, u_lam );

					u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
					u_qx   = _mm_hsub_pd( u_qx, u_qx );
					u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
					u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
					u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
					u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
					_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
					_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

					ptr_t     += 2;
					ptr_lam   += 2;
					ptr_lamt  += 2;
					ptr_dlam  += 2;
					ptr_t_inv += 2;

					ptr_db    += 2;
					ptr_Z     += 2;
					ptr_z     += 2;
					ptr_Zl    += 2;
					ptr_zl    += 2;
					
					ll++;

					}

				}
		
			// soft constraints on x
			// clean-up loop
			bs0 = nb-ii<4 ? nb-ii : 4 ; // either 0 ro 2 constraints to be done !!!

			if(ll<bs0)
				{

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp2  = _mm256_load_pd( &ptr_t[pnb+0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[pnb+0], v_tmp2 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lam2  = _mm256_load_pd( &ptr_lam[pnb+0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				_mm256_store_pd( &ptr_lamt[pnb+0], v_lamt2 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
				_mm256_store_pd( &ptr_dlam[pnb+0], v_dlam2 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
				v_zl0  = _mm256_load_pd( &ptr_z[0] );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
				v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
				v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
				v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
				v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
				_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
				_mm256_store_pd( &ptr_zl[0], v_zl0 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
				v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
				v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
				v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;

				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;
			
				ll   += 2;

				}
			ii += ll;
			}

		// soft constraints main loop
		for(; ii<nb-3; ii+=4)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[4] );
			v_tmp2  = _mm256_load_pd( &ptr_t[pnb+0] );
			v_tmp3  = _mm256_load_pd( &ptr_t[pnb+4] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
			v_tmp3  = _mm256_div_pd( v_ones, v_tmp3 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[4], v_tmp1 );
			_mm256_store_pd( &ptr_t_inv[pnb+0], v_tmp2 );
			_mm256_store_pd( &ptr_t_inv[pnb+4], v_tmp3 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[4] );
			v_lam2  = _mm256_load_pd( &ptr_lam[pnb+0] );
			v_lam3  = _mm256_load_pd( &ptr_lam[pnb+4] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
			v_lamt3 = _mm256_mul_pd( v_tmp3, v_lam3 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
			_mm256_store_pd( &ptr_lamt[pnb+0], v_lamt2 );
			_mm256_store_pd( &ptr_lamt[pnb+4], v_lamt3 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
			v_dlam3 = _mm256_mul_pd( v_tmp3, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[4], v_dlam1 );
			_mm256_store_pd( &ptr_dlam[pnb+0], v_dlam2 );
			_mm256_store_pd( &ptr_dlam[pnb+4], v_dlam3 );

			v_Qx0 = v_lamt0;
			v_Qx1 = v_lamt1;
			v_qx0  = _mm256_load_pd( &db[jj][2*ii+0] );
			v_qx1  = _mm256_load_pd( &db[jj][2*ii+4] );
			v_qx0  = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1  = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_dlam1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_lam1 );

			v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
			v_Zl1  = _mm256_load_pd( &ptr_Z[4] );
			v_zl0  = _mm256_load_pd( &ptr_z[0] );
			v_zl1  = _mm256_load_pd( &ptr_z[4] );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
			v_Zl1  = _mm256_add_pd( v_Zl1, v_Qx1 );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
			v_Zl1  = _mm256_add_pd( v_Zl1, v_lamt3 );
			v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
			v_Zl1  = _mm256_div_pd( v_ones, v_Zl1 );
			v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
			v_zl1  = _mm256_sub_pd( v_qx1, v_zl1 );
			v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
			v_zl1  = _mm256_add_pd( v_zl1, v_lam3 );
			v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
			v_zl1  = _mm256_add_pd( v_zl1, v_dlam3 );
			_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
			_mm256_store_pd( &ptr_Zl[4], v_Zl1 );
			_mm256_store_pd( &ptr_zl[0], v_zl0 );
			_mm256_store_pd( &ptr_zl[4], v_zl1 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
			v_tmp1 = _mm256_mul_pd( v_Qx1, v_Zl1 );
			v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
			v_tmp3 = _mm256_mul_pd( v_tmp1, v_zl1 );
			v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
			v_qx1  = _mm256_sub_pd( v_qx1, v_tmp3 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
			v_tmp1 = _mm256_mul_pd( v_Qx1, v_tmp1 );
			v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );
			v_Qx1  = _mm256_sub_pd( v_Qx1, v_tmp1 );

			v_tmp0 = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
			v_Qx1  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
			v_Qx0  = v_tmp0;
			v_tmp1 = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
			v_qx1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
			v_qx0  = v_tmp1;
			v_Qx0  = _mm256_hadd_pd( v_Qx0, v_Qx1 );
			v_qx0  = _mm256_hsub_pd( v_qx0, v_qx1 );
			v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
			v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
			v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
			_mm256_store_pd( &pd[jj][ii], v_tmp0 );
			_mm256_store_pd( &pl[jj][ii], v_tmp1 );

			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_t_inv += 8;

			ptr_db    += 8;
			ptr_Z     += 8;
			ptr_z     += 8;
			ptr_Zl    += 8;
			ptr_zl    += 8;

			}
		if(ii<nb)
			{
			bs0 = nb-ii;
			ll = 0;
			
			if(bs0>=2)
				{

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp2  = _mm256_load_pd( &ptr_t[pnb+0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[pnb+0], v_tmp2 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lam2  = _mm256_load_pd( &ptr_lam[pnb+0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				_mm256_store_pd( &ptr_lamt[pnb+0], v_lamt2 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
				_mm256_store_pd( &ptr_dlam[pnb+0], v_dlam2 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
				v_zl0  = _mm256_load_pd( &ptr_z[0] );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
				v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
				v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
				v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
				v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
				_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
				_mm256_store_pd( &ptr_zl[0], v_zl0 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
				v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
				v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
				v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;

				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;
			
				ll   += 2;
				bs0  -= 2;

				}
			
			if(bs0>0)
				{
				
				u_tmp0  = _mm_load_pd( &ptr_t[0] );
				u_tmp1  = _mm_load_pd( &ptr_t[pnb+0] );
				u_tmp0  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
				u_tmp1  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp1 );
				_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
				_mm_store_pd( &ptr_t_inv[pnb+0], u_tmp1 );
				u_lam0  = _mm_load_pd( &ptr_lam[0] );
				u_lam1  = _mm_load_pd( &ptr_lam[pnb+0] );
				u_lamt0 = _mm_mul_pd( u_tmp0, u_lam0 );
				u_lamt1 = _mm_mul_pd( u_tmp1, u_lam1 );
				_mm_store_pd( &ptr_lamt[0], u_lamt0 );
				_mm_store_pd( &ptr_lamt[pnb+0], u_lamt1 );
				u_dlam0 = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
				u_dlam1 = _mm_mul_pd( u_tmp1, _mm256_castpd256_pd128( v_sigma_mu ) );
				_mm_store_pd( &ptr_dlam[0], u_dlam0 );
				_mm_store_pd( &ptr_dlam[pnb+0], u_dlam1 );

				u_Qx   = u_lamt0;
				u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
				u_qx   = _mm_mul_pd( u_qx, u_lamt0 );
				u_qx   = _mm_add_pd( u_qx, u_dlam0 );
				u_qx   = _mm_add_pd( u_qx, u_lam0 );

				u_Zl   = _mm_load_pd( &ptr_Z[0] );
				u_zl   = _mm_load_pd( &ptr_z[0] );
				u_Zl   = _mm_add_pd( u_Zl, u_Qx );
				u_Zl   = _mm_add_pd( u_Zl, u_lamt1 );
				u_Zl   = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_Zl );
				u_zl   = _mm_sub_pd( u_qx, u_zl );
				u_zl   = _mm_add_pd( u_zl, u_lam1 );
				u_zl   = _mm_add_pd( u_zl, u_dlam1 );
				_mm_store_pd( &ptr_Zl[0], u_Zl );
				_mm_store_pd( &ptr_zl[0], u_zl );
				u_tmp0 = _mm_mul_pd( u_Qx, u_Zl );
				u_tmp1 = _mm_mul_pd( u_tmp0, u_zl );
				u_qx  = _mm_sub_pd( u_qx, u_tmp1 );
				u_tmp0 = _mm_mul_pd( u_Qx, u_tmp0 );
				u_Qx  = _mm_sub_pd( u_Qx, u_tmp0 );

				u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
				u_qx   = _mm_hsub_pd( u_qx, u_qx );
				u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
				u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
				u_tmp0 = _mm_add_sd( u_tmp0, u_Qx );
				u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
				_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

//				ptr_t     += 2;
//				ptr_lam   += 2;
//				ptr_lamt  += 2;
//				ptr_dlam  += 2;
//				ptr_t_inv += 2;

//				ptr_db    += 2;
//				ptr_Z     += 2;
//				ptr_z     += 2;
//				ptr_Zl    += 2;
//				ptr_zl    += 2;

				ll++;
				}

			ii += ll;
			}
		for( ; ii<nu+nx; ii++)
			{
			pd[jj][ii] = bd[jj][ii];
			pl[jj][ii] = bl[jj][ii];
			}
	
		}

	// last stage
	jj = N;

	ptr_t     = t[N]     + 2*nu;
	ptr_lam   = lam[N]   + 2*nu;
	ptr_lamt  = lamt[N]  + 2*nu;
	ptr_dlam  = dlam[N]  + 2*nu;
	ptr_t_inv  = t_inv[N] + 2*nu;
	ptr_db    = db[N]    + 2*nu;
	ptr_Z     = Z[N]     + 2*nu;
	ptr_z     = z[N]     + 2*nu;
	ptr_Zl    = Zl[N]    + 2*nu;
	ptr_zl    = zl[N]    + 2*nu;
	ptr_pd    = pd[N];
	ptr_pl    = pl[N];
	ptr_bd    = bd[N];
	ptr_bl    = bl[N];

	ii=4*(nu/4); // k1 supposed to be multiple of bs !!!!!!!!!! NO MORE !!!!!!!
	if(nh>nu) // there are hard state-constraints
		{
	
		if(ii<nu)
			{
			bs0 = nh-ii<4 ? nh-ii : 4 ;
			ll = nu-ii;
			if(ll%2==1)
				{
				u_tmp0 = _mm_load_pd( &ptr_t[0] );
				u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
				_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
				u_lam  = _mm_load_pd( &ptr_lam[0] );
				u_lamt = _mm_mul_pd( u_tmp0, u_lam );
				_mm_store_pd( &ptr_lamt[0], u_lamt );
				u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
				_mm_store_pd( &ptr_dlam[0], u_dlam );

				u_Qx   = u_lamt;
				u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
				u_qx   = _mm_mul_pd( u_qx, u_lamt );
				u_qx   = _mm_add_pd( u_qx, u_dlam );
				u_qx   = _mm_add_pd( u_qx, u_lam );

				u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
				u_qx   = _mm_hsub_pd( u_qx, u_qx );
				u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
				u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
				u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
				u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
				_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 2;
				ptr_lam   += 2;
				ptr_lamt  += 2;
				ptr_dlam  += 2;
				ptr_t_inv += 2;

				ptr_db    += 2;
				ptr_Z     += 2;
				ptr_z     += 2;
				ptr_Zl    += 2;
				ptr_zl    += 2;

				ll++;
				}
			if(ll<bs0)
				{
				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;
				
				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;

				ll += 2;
				}
			ii += ll;
			}

		// hard constraints on u and x
		for(; ii<nh-3; ii+=4)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[4] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 ); // store t_inv
			_mm256_store_pd( &ptr_t_inv[4], v_tmp1 ); // store t_inv
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[4] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[4], v_dlam1 );

			v_Qx0  = v_lamt0;
			v_Qx1  = v_lamt1;
			v_qx0  = _mm256_load_pd( &db[jj][2*ii+0] );
			v_qx1  = _mm256_load_pd( &db[jj][2*ii+4] );
			v_qx0  = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1  = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_dlam1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_lam1 );

			v_tmp0 = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
			v_Qx1  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
			v_Qx0  = v_tmp0;
			v_tmp1 = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
			v_qx1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
			v_qx0  = v_tmp1;
			v_Qx0  = _mm256_hadd_pd( v_Qx0, v_Qx1 );
			v_qx0  = _mm256_hsub_pd( v_qx0, v_qx1 );
			v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
			v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
			v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
			_mm256_store_pd( &pd[jj][ii], v_tmp0 );
			_mm256_store_pd( &pl[jj][ii], v_tmp1 );

			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_t_inv += 8;

			ptr_db    += 8;
			ptr_Z     += 8;
			ptr_z     += 8;
			ptr_Zl    += 8;
			ptr_zl    += 8;

			}
		if(ii<nh)
			{
			// clean-up loop
			bs0 = nh-ii;
			ll = 0;
			if(bs0>=2)
				{

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;
				
				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;
				
				ll   += 2;
				bs0  -= 2;

				}
			
			if(bs0>0)
				{
				if(nh<nb) // there are soft constraints afterwards
					{

					v_tmp0  = _mm256_load_pd( &ptr_t[0] );
					u_tmp1  = _mm_load_pd( &ptr_t[pnb+2] );
					v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
					u_tmp1  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp1 );
					_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
					_mm_store_pd( &ptr_t_inv[pnb+2], u_tmp1 );
					v_lam0  = _mm256_load_pd( &ptr_lam[0] );
					u_lam1  = _mm_load_pd( &ptr_lam[pnb+2] );
					v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
					u_lamt1 = _mm_mul_pd( u_tmp1, u_lam1 );
					_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
					_mm_store_pd( &ptr_lamt[pnb+2], u_lamt1 );
					v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
					u_dlam1 = _mm_mul_pd( u_tmp1, _mm256_castpd256_pd128( v_sigma_mu ) );
					_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
					_mm_store_pd( &ptr_dlam[pnb+2], u_dlam1 );

					v_Qx0   = v_lamt0;
					v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
					v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
					v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
					v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

					u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
					u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
					u_Zl   = _mm_load_pd( &ptr_Z[2] );
					u_zl   = _mm_load_pd( &ptr_z[2] );
					u_Zl   = _mm_add_pd( u_Zl, u_Qx );
					u_Zl   = _mm_add_pd( u_Zl, u_lamt1 );
					u_Zl   = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_Zl );
					u_zl   = _mm_sub_pd( u_qx, u_zl );
					u_zl   = _mm_add_pd( u_zl, u_lam1 );
					u_zl   = _mm_add_pd( u_zl, u_dlam1 );
					_mm_store_pd( &ptr_Zl[2], u_Zl );
					_mm_store_pd( &ptr_zl[2], u_zl );
					u_tmp0 = _mm_mul_pd( u_Qx, u_Zl );
					u_tmp1 = _mm_mul_pd( u_tmp0, u_zl );
					u_qx  = _mm_sub_pd( u_qx, u_tmp1 );
					u_tmp0 = _mm_mul_pd( u_Qx, u_tmp0 );
					u_Qx  = _mm_sub_pd( u_Qx, u_tmp0 );

					u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
					u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
					u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
					u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
					u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
					u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
					_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
					_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

					ptr_t     += 4;
					ptr_lam   += 4;
					ptr_lamt  += 4;
					ptr_dlam  += 4;
					ptr_t_inv += 4;

					ptr_db    += 4;
					ptr_Z     += 4;
					ptr_z     += 4;
					ptr_Zl    += 4;
					ptr_zl    += 4;
				
					ll   += 2;
					}
				else // no soft constraints afterward
					{

					u_tmp0 = _mm_load_pd( &ptr_t[0] );
					u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
					_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
					u_lam  = _mm_load_pd( &ptr_lam[0] );
					u_lamt = _mm_mul_pd( u_tmp0, u_lam );
					_mm_store_pd( &ptr_lamt[0], u_lamt );
					u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
					_mm_store_pd( &ptr_dlam[0], u_dlam );

					u_Qx   = u_lamt;
					u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
					u_qx   = _mm_mul_pd( u_qx, u_lamt );
					u_qx   = _mm_add_pd( u_qx, u_dlam );
					u_qx   = _mm_add_pd( u_qx, u_lam );

					u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
					u_qx   = _mm_hsub_pd( u_qx, u_qx );
					u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
					u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
					u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
					u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
					_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
					_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

					ptr_t     += 2;
					ptr_lam   += 2;
					ptr_lamt  += 2;
					ptr_dlam  += 2;
					ptr_t_inv += 2;

					ptr_db    += 2;
					ptr_Z     += 2;
					ptr_z     += 2;
					ptr_Zl    += 2;
					ptr_zl    += 2;
					
					ll++;

					}

				}
		
			// soft constraints on x
			// clean-up loop
			bs0 = nb-ii<4 ? nb-ii : 4 ; // either 0 ro 2 constraints to be done !!!

			if(ll<bs0)
				{

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp2  = _mm256_load_pd( &ptr_t[pnb+0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[pnb+0], v_tmp2 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lam2  = _mm256_load_pd( &ptr_lam[pnb+0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				_mm256_store_pd( &ptr_lamt[pnb+0], v_lamt2 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
				_mm256_store_pd( &ptr_dlam[pnb+0], v_dlam2 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
				v_zl0  = _mm256_load_pd( &ptr_z[0] );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
				v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
				v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
				v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
				v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
				_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
				_mm256_store_pd( &ptr_zl[0], v_zl0 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
				v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
				v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
				v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;

				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;
			
				ll   += 2;

				}
			ii += ll;
			}

		}
	else // there are not hard state-constraints
		{

		if(ii<nu)
			{
			bs0 = nb-ii<4 ? nb-ii : 4 ;
			ll = nu-ii;
			if(ll%2==1)
				{

				u_tmp0  = _mm_load_pd( &ptr_t[0] );
				u_tmp1  = _mm_load_pd( &ptr_t[pnb+0] );
				u_tmp0  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
				u_tmp1  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp1 );
				_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
				_mm_store_pd( &ptr_t_inv[pnb+0], u_tmp1 );
				u_lam0  = _mm_load_pd( &ptr_lam[0] );
				u_lam1  = _mm_load_pd( &ptr_lam[pnb+0] );
				u_lamt0 = _mm_mul_pd( u_tmp0, u_lam0 );
				u_lamt1 = _mm_mul_pd( u_tmp1, u_lam1 );
				_mm_store_pd( &ptr_lamt[0], u_lamt0 );
				_mm_store_pd( &ptr_lamt[pnb+0], u_lamt1 );
				u_dlam0 = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
				u_dlam1 = _mm_mul_pd( u_tmp1, _mm256_castpd256_pd128( v_sigma_mu ) );
				_mm_store_pd( &ptr_dlam[0], u_dlam0 );
				_mm_store_pd( &ptr_dlam[pnb+0], u_dlam1 );

				u_Qx   = u_lamt0;
				u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
				u_qx   = _mm_mul_pd( u_qx, u_lamt0 );
				u_qx   = _mm_add_pd( u_qx, u_dlam0 );
				u_qx   = _mm_add_pd( u_qx, u_lam0 );

				u_Zl   = _mm_load_pd( &ptr_Z[0] );
				u_zl   = _mm_load_pd( &ptr_z[0] );
				u_Zl   = _mm_add_pd( u_Zl, u_Qx );
				u_Zl   = _mm_add_pd( u_Zl, u_lamt1 );
				u_Zl   = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_Zl );
				u_zl   = _mm_sub_pd( u_qx, u_zl );
				u_zl   = _mm_add_pd( u_zl, u_lam1 );
				u_zl   = _mm_add_pd( u_zl, u_dlam1 );
				_mm_store_pd( &ptr_Zl[0], u_Zl );
				_mm_store_pd( &ptr_zl[0], u_zl );
				u_tmp0 = _mm_mul_pd( u_Qx, u_Zl );
				u_tmp1 = _mm_mul_pd( u_tmp0, u_zl );
				u_qx  = _mm_sub_pd( u_qx, u_tmp1 );
				u_tmp0 = _mm_mul_pd( u_Qx, u_tmp0 );
				u_Qx  = _mm_sub_pd( u_Qx, u_tmp0 );

				u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
				u_qx   = _mm_hsub_pd( u_qx, u_qx );
				u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
				u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
				u_tmp0 = _mm_add_sd( u_tmp0, u_Qx );
				u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
				_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 2;
				ptr_lam   += 2;
				ptr_lamt  += 2;
				ptr_dlam  += 2;
				ptr_t_inv += 2;

				ptr_db    += 2;
				ptr_Z     += 2;
				ptr_z     += 2;
				ptr_Zl    += 2;
				ptr_zl    += 2;

				ll++;
				}
			if(ll<bs0)
				{

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp2  = _mm256_load_pd( &ptr_t[pnb+0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[pnb+0], v_tmp2 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lam2  = _mm256_load_pd( &ptr_lam[pnb+0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				_mm256_store_pd( &ptr_lamt[pnb+0], v_lamt2 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
				_mm256_store_pd( &ptr_dlam[pnb+0], v_dlam2 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
				v_zl0  = _mm256_load_pd( &ptr_z[0] );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
				v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
				v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
				v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
				v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
				_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
				_mm256_store_pd( &ptr_zl[0], v_zl0 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
				v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
				v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
				v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;

				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;
			
				ll+=2;
				}

			ii += ll;
			}

		}

	// soft constraint main loop
	for(; ii<nb-3; ii+=4)
		{

		v_tmp0  = _mm256_load_pd( &ptr_t[0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[4] );
		v_tmp2  = _mm256_load_pd( &ptr_t[pnb+0] );
		v_tmp3  = _mm256_load_pd( &ptr_t[pnb+4] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
		v_tmp3  = _mm256_div_pd( v_ones, v_tmp3 );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
		_mm256_store_pd( &ptr_t_inv[4], v_tmp1 );
		_mm256_store_pd( &ptr_t_inv[pnb+0], v_tmp2 );
		_mm256_store_pd( &ptr_t_inv[pnb+4], v_tmp3 );
		v_lam0  = _mm256_load_pd( &ptr_lam[0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[4] );
		v_lam2  = _mm256_load_pd( &ptr_lam[pnb+0] );
		v_lam3  = _mm256_load_pd( &ptr_lam[pnb+4] );
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
		v_lamt3 = _mm256_mul_pd( v_tmp3, v_lam3 );
		_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
		_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
		_mm256_store_pd( &ptr_lamt[pnb+0], v_lamt2 );
		_mm256_store_pd( &ptr_lamt[pnb+4], v_lamt3 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
		v_dlam3 = _mm256_mul_pd( v_tmp3, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[4], v_dlam1 );
		_mm256_store_pd( &ptr_dlam[pnb+0], v_dlam2 );
		_mm256_store_pd( &ptr_dlam[pnb+4], v_dlam3 );

		v_Qx0 = v_lamt0;
		v_Qx1 = v_lamt1;
		v_qx0  = _mm256_load_pd( &db[jj][2*ii+0] );
		v_qx1  = _mm256_load_pd( &db[jj][2*ii+4] );
		v_qx0  = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1  = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_qx0  = _mm256_add_pd( v_qx0, v_dlam0 );
		v_qx1  = _mm256_add_pd( v_qx1, v_dlam1 );
		v_qx0  = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1  = _mm256_add_pd( v_qx1, v_lam1 );

		v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
		v_Zl1  = _mm256_load_pd( &ptr_Z[4] );
		v_zl0  = _mm256_load_pd( &ptr_z[0] );
		v_zl1  = _mm256_load_pd( &ptr_z[4] );
		v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
		v_Zl1  = _mm256_add_pd( v_Zl1, v_Qx1 );
		v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
		v_Zl1  = _mm256_add_pd( v_Zl1, v_lamt3 );
		v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
		v_Zl1  = _mm256_div_pd( v_ones, v_Zl1 );
		v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
		v_zl1  = _mm256_sub_pd( v_qx1, v_zl1 );
		v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
		v_zl1  = _mm256_add_pd( v_zl1, v_lam3 );
		v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
		v_zl1  = _mm256_add_pd( v_zl1, v_dlam3 );
		_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
		_mm256_store_pd( &ptr_Zl[4], v_Zl1 );
		_mm256_store_pd( &ptr_zl[0], v_zl0 );
		_mm256_store_pd( &ptr_zl[4], v_zl1 );
		v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
		v_tmp1 = _mm256_mul_pd( v_Qx1, v_Zl1 );
		v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
		v_tmp3 = _mm256_mul_pd( v_tmp1, v_zl1 );
		v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
		v_qx1  = _mm256_sub_pd( v_qx1, v_tmp3 );
		v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
		v_tmp1 = _mm256_mul_pd( v_Qx1, v_tmp1 );
		v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );
		v_Qx1  = _mm256_sub_pd( v_Qx1, v_tmp1 );

		v_tmp0 = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
		v_Qx1  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
		v_Qx0  = v_tmp0;
		v_tmp1 = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
		v_qx1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
		v_qx0  = v_tmp1;
		v_Qx0  = _mm256_hadd_pd( v_Qx0, v_Qx1 );
		v_qx0  = _mm256_hsub_pd( v_qx0, v_qx1 );
		v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
		v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
		v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
		v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
		_mm256_store_pd( &pd[jj][ii], v_tmp0 );
		_mm256_store_pd( &pl[jj][ii], v_tmp1 );

		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;
		ptr_db    += 8;
		ptr_Z     += 8;
		ptr_z     += 8;
		ptr_Zl    += 8;
		ptr_zl    += 8;

		}
	if(ii<nb)
		{
		bs0 = nb-ii;
		ll = 0;
		
		if(bs0>=2)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp2  = _mm256_load_pd( &ptr_t[pnb+0] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[pnb+0], v_tmp2 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lam2  = _mm256_load_pd( &ptr_lam[pnb+0] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[pnb+0], v_lamt2 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[pnb+0], v_dlam2 );

			v_Qx0   = v_lamt0;
			v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

			v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
			v_zl0  = _mm256_load_pd( &ptr_z[0] );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
			v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
			v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
			v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
			v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
			_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
			_mm256_store_pd( &ptr_zl[0], v_zl0 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
			v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
			v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
			v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );

			u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
			u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
			u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
			u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
			u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
			u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
			u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
			u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
			_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

			ptr_t     += 4;
			ptr_lam   += 4;
			ptr_lamt  += 4;
			ptr_dlam  += 4;
			ptr_t_inv += 4;

			ptr_db    += 4;
			ptr_Z     += 4;
			ptr_z     += 4;
			ptr_Zl    += 4;
			ptr_zl    += 4;
		
			ll   += 2;
			bs0  -= 2;

			}
		
		if(bs0>0)
			{
			
			u_tmp0  = _mm_load_pd( &ptr_t[0] );
			u_tmp1  = _mm_load_pd( &ptr_t[pnb+0] );
			u_tmp0  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
			u_tmp1  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp1 );
			_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
			_mm_store_pd( &ptr_t_inv[pnb+0], u_tmp1 );
			u_lam0  = _mm_load_pd( &ptr_lam[0] );
			u_lam1  = _mm_load_pd( &ptr_lam[pnb+0] );
			u_lamt0 = _mm_mul_pd( u_tmp0, u_lam0 );
			u_lamt1 = _mm_mul_pd( u_tmp1, u_lam1 );
			_mm_store_pd( &ptr_lamt[0], u_lamt0 );
			_mm_store_pd( &ptr_lamt[pnb+0], u_lamt1 );
			u_dlam0 = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
			u_dlam1 = _mm_mul_pd( u_tmp1, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam0 );
			_mm_store_pd( &ptr_dlam[pnb+0], u_dlam1 );

			u_Qx   = u_lamt0;
			u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
			u_qx   = _mm_mul_pd( u_qx, u_lamt0 );
			u_qx   = _mm_add_pd( u_qx, u_dlam0 );
			u_qx   = _mm_add_pd( u_qx, u_lam0 );

			u_Zl   = _mm_load_pd( &ptr_Z[0] );
			u_zl   = _mm_load_pd( &ptr_z[0] );
			u_Zl   = _mm_add_pd( u_Zl, u_Qx );
			u_Zl   = _mm_add_pd( u_Zl, u_lamt1 );
			u_Zl   = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_Zl );
			u_zl   = _mm_sub_pd( u_qx, u_zl );
			u_zl   = _mm_add_pd( u_zl, u_lam1 );
			u_zl   = _mm_add_pd( u_zl, u_dlam1 );
			_mm_store_pd( &ptr_Zl[0], u_Zl );
			_mm_store_pd( &ptr_zl[0], u_zl );
			u_tmp0 = _mm_mul_pd( u_Qx, u_Zl );
			u_tmp1 = _mm_mul_pd( u_tmp0, u_zl );
			u_qx  = _mm_sub_pd( u_qx, u_tmp1 );
			u_tmp0 = _mm_mul_pd( u_Qx, u_tmp0 );
			u_Qx  = _mm_sub_pd( u_Qx, u_tmp0 );

			u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
			u_qx   = _mm_hsub_pd( u_qx, u_qx );
			u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
			u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
			u_tmp0 = _mm_add_sd( u_tmp0, u_Qx );
			u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
			_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

//				ptr_t     += 2;
//				ptr_lam   += 2;
//				ptr_lamt  += 2;
//				ptr_dlam  += 2;
//				ptr_t_inv += 2;

//				ptr_db    += 2;
//				ptr_Z     += 2;
//				ptr_z     += 2;
//				ptr_Zl    += 2;
//				ptr_zl    += 2;

			ll++;
			}

		ii += ll;
		}
	for( ; ii<nu+nx; ii++)
		{
		pd[jj][ii] = bd[jj][ii];
		pl[jj][ii] = bl[jj][ii];
		}


	}



void d_update_hessian_diag_mpc(int N, int *nx, int *nu, int *nb, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **db)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	double 
		*ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Qx, *ptr_qx,
		*ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv;
	
	int ii, jj, bs0;

	int pnb;
	
	for(jj=0; jj<=N; jj++)
		{
		
		pnb  = bs*((nb[jj]+bs-1)/bs); // simd aligned number of box constraints

		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_tinv  = t_inv[jj];
		ptr_db    = db[jj];
		ptr_bd    = bd[jj];
		ptr_bl    = bl[jj];
		ptr_pd    = pd[jj];
		ptr_pl    = pl[jj];

		ii = 0;
		for(; ii<nb[jj]-3; ii+=4)
			{

			ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
			ptr_tinv[ii+pnb+0] = 1.0/ptr_t[ii+pnb+0];
			ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
			ptr_lamt[ii+pnb+0] = ptr_lam[ii+pnb+0]*ptr_tinv[ii+pnb+0];
			ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
			ptr_dlam[ii+pnb+0] = ptr_tinv[ii+pnb+0]*sigma_mu; // !!!!!
			ptr_pd[ii+0] = ptr_bd[ii+0] + ptr_lamt[ii+0] + ptr_lamt[ii+pnb+0];
			ptr_pl[ii+0] = ptr_bl[ii+0] + ptr_lam[ii+pnb+0] + ptr_lamt[ii+pnb+0]*ptr_db[ii+pnb+0] + ptr_dlam[ii+pnb+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0];

			ptr_tinv[ii+1] = 1.0/ptr_t[ii+1];
			ptr_tinv[ii+pnb+1] = 1.0/ptr_t[ii+pnb+1];
			ptr_lamt[ii+1] = ptr_lam[ii+1]*ptr_tinv[ii+1];
			ptr_lamt[ii+pnb+1] = ptr_lam[ii+pnb+1]*ptr_tinv[ii+pnb+1];
			ptr_dlam[ii+1] = ptr_tinv[ii+1]*sigma_mu; // !!!!!
			ptr_dlam[ii+pnb+1] = ptr_tinv[ii+pnb+1]*sigma_mu; // !!!!!
			ptr_pd[ii+1] = ptr_bd[ii+1] + ptr_lamt[ii+1] + ptr_lamt[ii+pnb+1];
			ptr_pl[ii+1] = ptr_bl[ii+1] + ptr_lam[ii+pnb+1] + ptr_lamt[ii+pnb+1]*ptr_db[ii+pnb+1] + ptr_dlam[ii+pnb+1] - ptr_lam[ii+1] - ptr_lamt[ii+1]*ptr_db[ii+1] - ptr_dlam[ii+1];

			ptr_tinv[ii+2] = 1.0/ptr_t[ii+2];
			ptr_tinv[ii+pnb+2] = 1.0/ptr_t[ii+pnb+2];
			ptr_lamt[ii+2] = ptr_lam[ii+2]*ptr_tinv[ii+2];
			ptr_lamt[ii+pnb+2] = ptr_lam[ii+pnb+2]*ptr_tinv[ii+pnb+2];
			ptr_dlam[ii+2] = ptr_tinv[ii+2]*sigma_mu; // !!!!!
			ptr_dlam[ii+pnb+2] = ptr_tinv[ii+pnb+2]*sigma_mu; // !!!!!
			ptr_pd[ii+2] = ptr_bd[ii+2] + ptr_lamt[ii+2] + ptr_lamt[ii+pnb+2];
			ptr_pl[ii+2] = ptr_bl[ii+2] + ptr_lam[ii+pnb+2] + ptr_lamt[ii+pnb+2]*ptr_db[ii+pnb+2] + ptr_dlam[ii+pnb+2] - ptr_lam[ii+2] - ptr_lamt[ii+2]*ptr_db[ii+2] - ptr_dlam[ii+2];

			ptr_tinv[ii+3] = 1.0/ptr_t[ii+3];
			ptr_tinv[ii+pnb+3] = 1.0/ptr_t[ii+pnb+3];
			ptr_lamt[ii+3] = ptr_lam[ii+3]*ptr_tinv[ii+3];
			ptr_lamt[ii+pnb+3] = ptr_lam[ii+pnb+3]*ptr_tinv[ii+pnb+3];
			ptr_dlam[ii+3] = ptr_tinv[ii+3]*sigma_mu; // !!!!!
			ptr_dlam[ii+pnb+3] = ptr_tinv[ii+pnb+3]*sigma_mu; // !!!!!
			ptr_pd[ii+3] = ptr_bd[ii+3] + ptr_lamt[ii+3] + ptr_lamt[ii+pnb+3];
			ptr_pl[ii+3] = ptr_bl[ii+3] + ptr_lam[ii+pnb+3] + ptr_lamt[ii+pnb+3]*ptr_db[ii+pnb+3] + ptr_dlam[ii+pnb+3] - ptr_lam[ii+3] - ptr_lamt[ii+3]*ptr_db[ii+3] - ptr_dlam[ii+3];

			}
		for(; ii<nb[jj]; ii++)
			{

			ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
			ptr_tinv[ii+pnb+0] = 1.0/ptr_t[ii+pnb+0];
			ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
			ptr_lamt[ii+pnb+0] = ptr_lam[ii+pnb+0]*ptr_tinv[ii+pnb+0];
			ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
			ptr_dlam[ii+pnb+0] = ptr_tinv[ii+pnb+0]*sigma_mu; // !!!!!
			ptr_pd[ii] = ptr_bd[ii] + ptr_lamt[ii+0] + ptr_lamt[ii+pnb+0];
			ptr_pl[ii] = ptr_bl[ii] + ptr_lam[ii+pnb+0] + ptr_lamt[ii+pnb+0]*ptr_db[ii+pnb+0] + ptr_dlam[ii+pnb+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0];

			}
		//for( ; ii<nu[jj]+nx[jj]; ii++)
		//	{
		//	ptr_pd[ii] = ptr_bd[ii];
		//	ptr_pl[ii] = ptr_bl[ii];
		//	}
	
		}

	}



void d_update_gradient_hard_mpc_tv(int N, int *nx, int *nu, int *nb, int *ng, double sigma_mu, double **dt, double **dlam, double **t_inv, double **pl2, double **qx)
	{

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	int ii, jj;

	int nb0, pnb, ng0, png;

	double
		*ptr_dlam, *ptr_t_inv, *ptr_dt, *ptr_pl2, *ptr_qx;

	for(jj=0; jj<=N; jj++)
		{

		nb0 = nb[jj];
		pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints
		ng0 = ng[jj];
		png  = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints

		ptr_dlam  = dlam[jj];
		ptr_dt    = dt[jj];
		ptr_t_inv = t_inv[jj];
		ptr_pl2   = pl2[jj];
		ptr_qx    = qx[jj];

		for(ii=0; ii<nb0; ii++)
			{
			ptr_dlam[ii]     = ptr_t_inv[ii]    *(sigma_mu - ptr_dlam[ii]*ptr_dt[ii]);
			ptr_dlam[pnb+ii] = ptr_t_inv[pnb+ii]*(sigma_mu - ptr_dlam[pnb+ii]*ptr_dt[pnb+ii]);
			ptr_pl2[ii] += ptr_dlam[pnb+ii] - ptr_dlam[ii];
			}

		for(ii=2*pnb; ii<2*pnb+ng0; ii++)
			{
			ptr_dlam[ii]     = ptr_t_inv[ii]    *(sigma_mu - ptr_dlam[ii]*ptr_dt[ii]);
			ptr_dlam[png+ii] = ptr_t_inv[png+ii]*(sigma_mu - ptr_dlam[png+ii]*ptr_dt[png+ii]);
			ptr_qx[ii] += ptr_dlam[png+ii] - ptr_dlam[ii];
			}

		}

	}




void d_update_gradient_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, double sigma_mu, double **dt, double **dlam, double **t_inv, double **pl2, double **qx)
	{

	const int nbu = nu<nb ? nu : nb ;
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	//const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box and soft constraints
	const int pnb  = bs*((nb+bs-1)/bs); // simd aligned number of box and soft constraints
	const int png  = bs*((ng+bs-1)/bs); // simd aligned number of box and soft constraints
	const int pngN = bs*((ngN+bs-1)/bs); // simd aligned number of box and soft constraints at last stage

	int ii, jj;

	double
		*ptr_dlam, *ptr_t_inv, *ptr_dt, *ptr_pl2, *ptr_qx;

	// first stage
	jj = 0;

	ptr_dlam  = dlam[jj];
	ptr_dt    = dt[jj];
	ptr_t_inv = t_inv[jj];
	ptr_pl2   = pl2[jj];

	for(ii=0; ii<nbu; ii++)
		{
		ptr_dlam[ii]     = ptr_t_inv[ii]    *(sigma_mu - ptr_dlam[ii]*ptr_dt[ii]); // !!!!!
		ptr_dlam[pnb+ii] = ptr_t_inv[pnb+ii]*(sigma_mu - ptr_dlam[pnb+ii]*ptr_dt[pnb+ii]); // !!!!!
		ptr_pl2[ii] += ptr_dlam[pnb+ii] - ptr_dlam[ii];
		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{

		ptr_dlam  = dlam[jj];
		ptr_dt    = dt[jj];
		ptr_t_inv = t_inv[jj];
		ptr_pl2   = pl2[jj];

		for(ii=0; ii<nb; ii++)
			{
			ptr_dlam[ii]     = ptr_t_inv[ii]    *(sigma_mu - ptr_dlam[ii]*ptr_dt[ii]); // !!!!!
			ptr_dlam[pnb+ii] = ptr_t_inv[pnb+ii]*(sigma_mu - ptr_dlam[pnb+ii]*ptr_dt[pnb+ii]); // !!!!!
			ptr_pl2[ii] += ptr_dlam[pnb+ii] - ptr_dlam[ii];
			}
		}

	// last stages
	jj = N;

	ptr_dlam  = dlam[jj];
	ptr_dt    = dt[jj];
	ptr_t_inv = t_inv[jj];
	ptr_pl2   = pl2[jj];

	for(ii=nu; ii<nb; ii++)
		{
		ptr_dlam[ii]     = ptr_t_inv[ii]    *(sigma_mu - ptr_dlam[ii]*ptr_dt[ii]); // !!!!!
		ptr_dlam[pnb+ii] = ptr_t_inv[pnb+ii]*(sigma_mu - ptr_dlam[pnb+ii]*ptr_dt[pnb+ii]); // !!!!!
		ptr_pl2[ii] += ptr_dlam[pnb+ii] - ptr_dlam[ii];
		}
	
	// general constraints
	if(ng>0)
		{

		for(jj=0; jj<N; jj++)
			{

			ptr_dlam  = dlam[jj];
			ptr_dt    = dt[jj];
			ptr_t_inv = t_inv[jj];
			ptr_pl2   = pl2[jj];
			ptr_qx    = qx[jj];

			for(ii=2*pnb; ii<2*pnb+ng; ii++)
				{
				ptr_dlam[ii]     = ptr_t_inv[ii]    *(sigma_mu - ptr_dlam[ii]*ptr_dt[ii]);
				ptr_dlam[png+ii] = ptr_t_inv[png+ii]*(sigma_mu - ptr_dlam[png+ii]*ptr_dt[png+ii]);
				ptr_qx[ii] = ptr_dlam[png+ii] - ptr_dlam[ii];
				}

			}

		}
	if(ngN>0)
		{

		ptr_dlam  = dlam[N];
		ptr_dt    = dt[N];
		ptr_t_inv = t_inv[N];
		ptr_pl2   = pl2[N];
		ptr_qx    = qx[N];

		for(ii=2*pnb; ii<2*pnb+ngN; ii++)
			{
			ptr_dlam[ii]     = ptr_t_inv[ii]    *(sigma_mu - ptr_dlam[ii]*ptr_dt[ii]);
			ptr_dlam[pngN+ii] = ptr_t_inv[pngN+ii]*(sigma_mu - ptr_dlam[pngN+ii]*ptr_dt[pngN+ii]);
			ptr_qx[ii] = ptr_dlam[pngN+ii] - ptr_dlam[ii];
			}

		}

	}



void d_update_gradient_soft_mpc(int N, int nx, int nu, int nh, int ns, double sigma_mu, double **dt, double **dlam, double **t_inv, double **lamt, double **pl2, double **Zl, double **zl)
	{

	int nb = nh + ns;

	int nhu = nu<nh ? nu : nh ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	int ii, jj;
	
	static double Qx[2] = {};
	static double qx[2] = {};


	// first stage
	jj = 0;
	for(ii=0; ii<2*nhu; ii+=2)
		{
		dlam[0][ii+0] = t_inv[0][ii+0]*(sigma_mu - dlam[0][ii+0]*dt[0][ii+0]); // !!!!!
		dlam[0][ii+1] = t_inv[0][ii+1]*(sigma_mu - dlam[0][ii+1]*dt[0][ii+1]); // !!!!!
		pl2[0][ii/2] += dlam[0][ii+1] - dlam[0][ii+0];
		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{
		ii=0;
		for(; ii<2*nh; ii+=2)
			{
			dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma_mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
			dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma_mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
			pl2[jj][ii/2] += dlam[jj][ii+1] - dlam[jj][ii+0];
			}
		for(; ii<2*nb; ii+=2)
			{
			dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma_mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
			dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma_mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
			dlam[jj][pnb+ii+0] = t_inv[jj][pnb+ii+0]*(sigma_mu - dlam[jj][pnb+ii+0]*dt[jj][pnb+ii+0]); // !!!!!
			dlam[jj][pnb+ii+1] = t_inv[jj][pnb+ii+1]*(sigma_mu - dlam[jj][pnb+ii+1]*dt[jj][pnb+ii+1]); // !!!!!
			Qx[0] = lamt[jj][ii+0];
			Qx[1] = lamt[jj][ii+1];
			//qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
			//qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
			qx[0] = dlam[jj][ii+0];
			qx[1] = dlam[jj][ii+1];
			//ptr_zl[0] = ptr_z[0] + qx[0] + ptr_lam[pnb+0] + ptr_dlam[pnb+0];
			//ptr_zl[1] = ptr_z[1] + qx[1] + ptr_lam[pnb+1] + ptr_dlam[pnb+1];
			zl[jj][ii+0] += qx[0] + dlam[jj][pnb+ii+0];
			zl[jj][ii+1] += qx[1] + dlam[jj][pnb+ii+1];
			//qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
			//qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
			qx[0] = qx[0] - Qx[0]*(qx[0] + dlam[jj][pnb+ii+0])*Zl[jj][ii+0]; // update this before Qx !!!!!!!!!!!
			qx[1] = qx[1] - Qx[1]*(qx[1] + dlam[jj][pnb+ii+1])*Zl[jj][ii+1]; // update this before Qx !!!!!!!!!!!
			pl2[jj][ii/2] += qx[1] - qx[0];
			}
		}

	// last stages
	jj = N;
	ii=2*nu;
	for(; ii<2*nh; ii+=2)
		{
		dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma_mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
		dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma_mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
		pl2[jj][ii/2] += dlam[jj][ii+1] - dlam[jj][ii+0];
		}
	for(; ii<2*nb; ii+=2)
		{
		dlam[N][ii+0] = t_inv[N][ii+0]*(sigma_mu - dlam[N][ii+0]*dt[N][ii+0]); // !!!!!
		dlam[N][ii+1] = t_inv[N][ii+1]*(sigma_mu - dlam[N][ii+1]*dt[N][ii+1]); // !!!!!
		dlam[N][pnb+ii+0] = t_inv[N][pnb+ii+0]*(sigma_mu - dlam[N][pnb+ii+0]*dt[N][pnb+ii+0]); // !!!!!
		dlam[N][pnb+ii+1] = t_inv[N][pnb+ii+1]*(sigma_mu - dlam[N][pnb+ii+1]*dt[N][pnb+ii+1]); // !!!!!
		Qx[0] = lamt[N][ii+0];
		Qx[1] = lamt[N][ii+1];
		//qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
		//qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
		qx[0] = dlam[N][ii+0];
		qx[1] = dlam[N][ii+1];
		//ptr_zl[0] = ptr_z[0] + qx[0] + ptr_lam[pnb+0] + ptr_dlam[pnb+0];
		//ptr_zl[1] = ptr_z[1] + qx[1] + ptr_lam[pnb+1] + ptr_dlam[pnb+1];
		zl[N][ii+0] += qx[0] + dlam[N][pnb+ii+0];
		zl[N][ii+1] += qx[1] + dlam[N][pnb+ii+1];
		//qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
		//qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
		qx[0] = qx[0] - Qx[0]*(qx[0] + dlam[N][pnb+ii+0])*Zl[N][ii+0]; // update this before Qx !!!!!!!!!!!
		qx[1] = qx[1] - Qx[1]*(qx[1] + dlam[N][pnb+ii+1])*Zl[N][ii+1]; // update this before Qx !!!!!!!!!!!
		pl2[N][ii/2] += qx[1] - qx[0];
		}

	}



void d_update_gradient_diag_mpc(int N, int *nx, int *nu, int *nb, double sigma_mu, double **dt, double **dlam, double **t_inv, double **pl2)
	{

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	int ii, jj;

	int pnb;

	double
		*ptr_dlam, *ptr_t_inv, *ptr_dt, *ptr_pl2, *ptr_qx;

	for(jj=0; jj<=N; jj++)
		{

		pnb  = bs*((nb[jj]+bs-1)/bs); // simd aligned number of box and soft constraints

		ptr_dlam  = dlam[jj];
		ptr_dt    = dt[jj];
		ptr_t_inv = t_inv[jj];
		ptr_pl2   = pl2[jj];

		for(ii=0; ii<nb[jj]; ii++)
			{
			ptr_dlam[ii]     = ptr_t_inv[ii]    *(sigma_mu - ptr_dlam[ii]*ptr_dt[ii]); // !!!!!
			ptr_dlam[pnb+ii] = ptr_t_inv[pnb+ii]*(sigma_mu - ptr_dlam[pnb+ii]*ptr_dt[pnb+ii]); // !!!!!
			ptr_pl2[ii] += ptr_dlam[pnb+ii] - ptr_dlam[ii];
			}
		}

	}



void d_compute_alpha_hard_mpc_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **pDCt, double **db)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	__m256
		t_sign, t_ones, t_zeros,
		t_mask0, t_mask1,
		t_lam, t_dlam, t_t, t_dt,
		t_tmp0, t_tmp1,
		t_alpha0, t_alpha1;

	__m128
		s_sign, s_ones, s_mask, s_mask0, s_mask1, s_zeros,
		s_lam, s_dlam, s_t, s_dt, s_tmp0, s_tmp1, s_alpha0, s_alpha1;
	
	__m256d
		v_sign, v_alpha, v_mask, v_left,
		v_temp0, v_dt0, v_dux, v_db0, v_dlam0, v_lamt0, v_t0, v_lam0,
		v_temp1, v_dt1, v_db1, v_dlam1, v_lamt1, v_t1, v_lam1;
	
	__m128d
		u_sign, u_dux, u_alpha,
		u_dt0, u_temp0, u_db0, u_dlam0, u_lamt0, u_t0, u_lam0,
		u_dt1, u_temp1, u_db1, u_dlam1, u_lamt1, u_t1, u_lam1;
	
	__m256i
		i_mask;
	
	int nu0, nx0, nb0, pnb, ng0, png, cng;

	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );
	u_sign = _mm_loaddup_pd( (double *) &long_sign );

	int int_sign = 0x80000000;
	s_sign = _mm_broadcast_ss( (float *) &int_sign );
	t_sign = _mm256_broadcast_ss( (float *) &int_sign );
	
	s_ones  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_zeros = _mm_setzero_ps( );

	s_alpha0 = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );

	t_ones  = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_zeros = _mm256_setzero_ps( );

	t_alpha0 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_alpha1 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	
	double ll_left;

	static double d_mask[4]  = {0.5, 1.5, 2.5, 3.5};

	static double dux_tmp[4] = {};

	double alpha = ptr_alpha[0];

	double
		*ptr_db, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lamt, *ptr_lam, *ptr_dlam;
	
	int
		*ptr_idxb;
	
	int ii, jj, ll;

	for(jj=0; jj<=N; jj++)
		{

		nb0 = nb[jj];
		pnb = (nb0+bs-1)/bs*bs;
		ng0 = ng[jj];

		ptr_db   = db[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lamt = lamt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];
		ptr_idxb = idxb[jj];

		// box constraints
		ll = 0;
		for(; ll<nb0-3; ll+=4)
			{
			//v_dux   = _mm256_load_pd( &ptr_dux[ll] );
			u_temp0 = _mm_load_sd( &ptr_dux[ptr_idxb[ll+0]] );
			u_temp1 = _mm_load_sd( &ptr_dux[ptr_idxb[ll+2]] );
			u_temp0 = _mm_loadh_pd( u_temp0, &ptr_dux[ptr_idxb[ll+1]] );
			u_temp1 = _mm_loadh_pd( u_temp1, &ptr_dux[ptr_idxb[ll+3]] );
			v_dux   = _mm256_castpd128_pd256( u_temp0 );
			v_dux   = _mm256_insertf128_pd( v_dux, u_temp1, 0x1 );
			v_db0   = _mm256_load_pd( &ptr_db[ll] );
			v_db1   = _mm256_load_pd( &ptr_db[pnb+ll] );
			v_db1   = _mm256_xor_pd( v_db1, v_sign );
			v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
			v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
			v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
			_mm256_store_pd( &ptr_dt[ll], v_dt0 );
			_mm256_store_pd( &ptr_dt[pnb+ll], v_dt1 );

			v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
			v_lamt1 = _mm256_load_pd( &ptr_lamt[pnb+ll] );
			v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
			v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
			_mm256_store_pd( &ptr_dlam[ll], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[pnb+ll], v_dlam1 );

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
			t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
			t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_t      = _mm256_xor_ps( t_t, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp1   = _mm256_div_ps( t_t, t_dt );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
			t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

			}
		if(ll<nb0)
			{

			for(ii=0; ii<nb0-ll; ii++) dux_tmp[ii] = ptr_dux[ptr_idxb[ll+ii]];

			ll_left = nb0 - ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_db0   = _mm256_load_pd( &ptr_db[ll] );
			v_db1   = _mm256_load_pd( &ptr_db[pnb+ll] );
			v_db1   = _mm256_xor_pd( v_db1, v_sign );
			v_dux   = _mm256_load_pd( &dux_tmp[0] );
			v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
			v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
			v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
			_mm256_maskstore_pd( &ptr_dt[ll], i_mask, v_dt0 );
			_mm256_maskstore_pd( &ptr_dt[pnb+ll], i_mask, v_dt1 );

			v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
			v_lamt1 = _mm256_load_pd( &ptr_lamt[pnb+ll] );
			v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
			v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
			_mm256_maskstore_pd( &ptr_dlam[ll], i_mask, v_dlam0 );
			_mm256_maskstore_pd( &ptr_dlam[pnb+ll], i_mask, v_dlam1 );

			if(ll<nb0-2) // 3 left
				{

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
				t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
				t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_t      = _mm256_xor_ps( t_t, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp1   = _mm256_div_ps( t_t, t_dt );
				t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
				t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
				t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

				}
			else // 1 or 2 left
				{

				s_mask   = _mm256_cvtpd_ps( v_mask );
				s_mask   = _mm_shuffle_ps( s_mask, s_mask, 0x44 );
				t_mask1  = _mm256_permute2f128_ps( _mm256_castps128_ps256( s_mask ), _mm256_castps128_ps256( s_mask ), 0x20 );
				t_mask1  = _mm256_cmp_ps( t_mask1, t_zeros, 0x01 );

				v_dt0    = _mm256_permute2f128_pd( v_dt0, v_dt1, 0x20 );
				v_t0     = _mm256_permute2f128_pd( v_t0, v_t1, 0x20 );
				v_dlam0  = _mm256_permute2f128_pd( v_dlam0, v_dlam1, 0x20 );
				v_lam0   = _mm256_permute2f128_pd( v_lam0, v_lam1, 0x20 );

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
				t_mask0  = _mm256_and_ps( t_mask0, t_mask1 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

				}
			}

		// general constraints
		if(ng0>0)
			{

			nu0 = nu[jj];
			nx0 = nx[jj];
			png = (ng0+bs-1)/bs*bs;
			cng = (ng0+ncl-1)/ncl*ncl;

			dgemv_t_lib(nx0+nu0, ng0, pDCt[jj], cng, ptr_dux, 0, ptr_dt+2*pnb, ptr_dt+2*pnb);

			for(ll=2*pnb; ll<2*pnb+ng0-3; ll+=4)
				{
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_xor_pd( v_dt0, v_sign );
				v_db0   = _mm256_load_pd( &ptr_db[ll] );
				v_db1   = _mm256_load_pd( &ptr_db[png+ll] );
				v_dt0   = _mm256_sub_pd ( v_dt0, v_db0 );
				v_dt1   = _mm256_sub_pd ( v_dt1, v_db1 );
				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
				v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
				v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
				_mm256_store_pd( &ptr_dt[ll], v_dt0 );
				_mm256_store_pd( &ptr_dt[png+ll], v_dt1 );

				v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
				v_lamt1 = _mm256_load_pd( &ptr_lamt[png+ll] );
				v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
				v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
				_mm256_store_pd( &ptr_dlam[ll], v_dlam0 );
				_mm256_store_pd( &ptr_dlam[png+ll], v_dlam1 );

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
				t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
				t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_t      = _mm256_xor_ps( t_t, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp1   = _mm256_div_ps( t_t, t_dt );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
				t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

				}
			if(ll<2*pnb+ng0)
				{

				ll_left = 2*pnb + ng0 - ll;
				v_left  = _mm256_broadcast_sd( &ll_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				v_mask  = _mm256_sub_pd( v_mask, v_left );
				i_mask  = _mm256_castpd_si256( v_mask );

				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_xor_pd( v_dt0, v_sign );
				v_db0   = _mm256_load_pd( &ptr_db[ll] );
				v_db1   = _mm256_load_pd( &ptr_db[png+ll] );
				v_dt0   = _mm256_sub_pd ( v_dt0, v_db0 );
				v_dt1   = _mm256_sub_pd ( v_dt1, v_db1 );
				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
				v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
				v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
				_mm256_maskstore_pd( &ptr_dt[ll], i_mask, v_dt0 );
				_mm256_maskstore_pd( &ptr_dt[png+ll], i_mask, v_dt1 );

				v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
				v_lamt1 = _mm256_load_pd( &ptr_lamt[png+ll] );
				v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
				v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
				_mm256_maskstore_pd( &ptr_dlam[ll], i_mask, v_dlam0 );
				_mm256_maskstore_pd( &ptr_dlam[png+ll], i_mask, v_dlam1 );

				if(ll<2*pnb+ng0-2) // 3 left
					{

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
					t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
					t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_t      = _mm256_xor_ps( t_t, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp1   = _mm256_div_ps( t_t, t_dt );
					t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
					t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
					t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

					}
				else // 1 or 2 left
					{

					s_mask   = _mm256_cvtpd_ps( v_mask );
					s_mask   = _mm_shuffle_ps( s_mask, s_mask, 0x44 );
					t_mask1  = _mm256_permute2f128_ps( _mm256_castps128_ps256( s_mask ), _mm256_castps128_ps256( s_mask ), 0x20 );
					t_mask1  = _mm256_cmp_ps( t_mask1, t_zeros, 0x01 );

					v_dt0    = _mm256_permute2f128_pd( v_dt0, v_dt1, 0x20 );
					v_t0     = _mm256_permute2f128_pd( v_t0, v_t1, 0x20 );
					v_dlam0  = _mm256_permute2f128_pd( v_dlam0, v_dlam1, 0x20 );
					v_lam0   = _mm256_permute2f128_pd( v_lam0, v_lam1, 0x20 );

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
					t_mask0  = _mm256_and_ps( t_mask0, t_mask1 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

					}
				}
			}

		}		

	// reduce alpha
	t_alpha0 = _mm256_min_ps( t_alpha0, t_alpha1 );
	s_alpha0 = _mm256_extractf128_ps( t_alpha0, 0x1 );
//	s_alpha1 = _mm256_extractf128_ps( t_alpha0, 0x1 );
//	s_alpha0  = _mm_min_ps( s_alpha0 , s_alpha1 );
	s_alpha1 = _mm256_castps256_ps128( t_alpha0 );
	s_alpha0 = _mm_min_ps( s_alpha0, s_alpha1 );
	
	v_alpha = _mm256_cvtps_pd( s_alpha0 );
	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );
	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );
	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
	_mm_store_sd( &alpha, u_alpha );

	
	ptr_alpha[0] = alpha;

	return;
	
	}


void d_compute_alpha_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **pDCt, double **db)
	{
	
	const int nbu = nu<nb ? nu : nb ;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	//const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box and soft constraints
	const int pnb  = bs*((nb+bs-1)/bs); // simd aligned number of box and soft constraints
	const int png  = bs*((ng+bs-1)/bs); // simd aligned number of box and soft constraints
	const int pngN = bs*((ngN+bs-1)/bs); // simd aligned number of box and soft constraints at last stage
	const int cng  = ncl*((ng+ncl-1)/ncl); // simd aligned number of box and soft constraints
	const int cngN = ncl*((ngN+ncl-1)/ncl); // simd aligned number of box and soft constraints at last stage

	__m256
		t_sign, t_ones, t_zeros,
		t_mask0, t_mask1,
		t_lam, t_dlam, t_t, t_dt,
		t_tmp0, t_tmp1,
		t_alpha0, t_alpha1;

	__m128
		s_sign, s_ones, s_mask, s_mask0, s_mask1, s_zeros,
		s_lam, s_dlam, s_t, s_dt, s_tmp0, s_tmp1, s_alpha0, s_alpha1;
	
	__m256d
		v_sign, v_alpha, v_mask, v_left,
		v_temp0, v_dt0, v_dux, v_db0, v_dlam0, v_lamt0, v_t0, v_lam0,
		v_temp1, v_dt1, v_db1, v_dlam1, v_lamt1, v_t1, v_lam1;
	
	__m128d
		u_sign, u_dux, u_alpha,
		u_dt0, u_temp0, u_db0, u_dlam0, u_lamt0, u_t0, u_lam0,
		u_dt1, u_temp1, u_db1, u_dlam1, u_lamt1, u_t1, u_lam1;
	
	__m256i
		i_mask;

	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );
	u_sign = _mm_loaddup_pd( (double *) &long_sign );

	int int_sign = 0x80000000;
	s_sign = _mm_broadcast_ss( (float *) &int_sign );
	t_sign = _mm256_broadcast_ss( (float *) &int_sign );
	
	s_ones  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_zeros = _mm_setzero_ps( );

	s_alpha0 = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );

	t_ones  = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_zeros = _mm256_setzero_ps( );

	t_alpha0 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_alpha1 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	
	double ll_left;

	static double d_mask[4]  = {0.5, 1.5, 2.5, 3.5};

	double alpha = ptr_alpha[0];

	double
		*ptr_db, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lamt, *ptr_lam, *ptr_dlam;
	
	int jj, ll;

	// first stage
	jj = 0;

	ptr_db   = db[jj];
	ptr_dux  = dux[jj];
	ptr_t    = t[jj];
	ptr_dt   = dt[jj];
	ptr_lamt = lamt[jj];
	ptr_lam  = lam[jj];
	ptr_dlam = dlam[jj];

	ll = 0;
	for(; ll<nbu-3; ll+=4)
		{

		v_db0   = _mm256_load_pd( &ptr_db[ll] );
		v_db1   = _mm256_load_pd( &ptr_db[pnb+ll] );
		v_db1   = _mm256_xor_pd( v_db1, v_sign );
		v_dux   = _mm256_load_pd( &ptr_dux[ll] );
		v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
		v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
		v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
		_mm256_store_pd( &ptr_dt[ll], v_dt0 );
		_mm256_store_pd( &ptr_dt[pnb+ll], v_dt1 );

		v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
		v_lamt1 = _mm256_load_pd( &ptr_lamt[pnb+ll] );
		v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
		v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
		v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
		v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
		v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
		_mm256_store_pd( &ptr_dlam[ll], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[pnb+ll], v_dlam1 );

		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
		t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
		t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_t      = _mm256_xor_ps( t_t, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp1   = _mm256_div_ps( t_t, t_dt );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
		t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

		}
	if(ll<nbu)
		{

		ll_left = nbu - ll;
		v_left  = _mm256_broadcast_sd( &ll_left );
		v_mask  = _mm256_loadu_pd( d_mask );
		v_mask  = _mm256_sub_pd( v_mask, v_left );
		i_mask  = _mm256_castpd_si256( v_mask );

		v_db0   = _mm256_load_pd( &ptr_db[ll] );
		v_db1   = _mm256_load_pd( &ptr_db[pnb+ll] );
		v_db1   = _mm256_xor_pd( v_db1, v_sign );
		v_dux   = _mm256_load_pd( &ptr_dux[ll] );
		v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
		v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
		v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
		_mm256_maskstore_pd( &ptr_dt[ll], i_mask, v_dt0 );
		_mm256_maskstore_pd( &ptr_dt[pnb+ll], i_mask, v_dt1 );

		v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
		v_lamt1 = _mm256_load_pd( &ptr_lamt[pnb+ll] );
		v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
		v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
		v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
		v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
		v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
		_mm256_maskstore_pd( &ptr_dlam[ll], i_mask, v_dlam0 );
		_mm256_maskstore_pd( &ptr_dlam[pnb+ll], i_mask, v_dlam1 );

		if(ll<nbu-2) // 3 left
			{

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
			t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
			t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_t      = _mm256_xor_ps( t_t, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp1   = _mm256_div_ps( t_t, t_dt );
			t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
			t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
			t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

			}
		else // 1 or 2 left
			{

			s_mask   = _mm256_cvtpd_ps( v_mask );
			s_mask   = _mm_shuffle_ps( s_mask, s_mask, 0x44 );
			t_mask1  = _mm256_permute2f128_ps( _mm256_castps128_ps256( s_mask ), _mm256_castps128_ps256( s_mask ), 0x20 );
			t_mask1  = _mm256_cmp_ps( t_mask1, t_zeros, 0x01 );

			v_dt0    = _mm256_permute2f128_pd( v_dt0, v_dt1, 0x20 );
			v_t0     = _mm256_permute2f128_pd( v_t0, v_t1, 0x20 );
			v_dlam0  = _mm256_permute2f128_pd( v_dlam0, v_dlam1, 0x20 );
			v_lam0   = _mm256_permute2f128_pd( v_lam0, v_lam1, 0x20 );

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
			t_mask0  = _mm256_and_ps( t_mask0, t_mask1 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

			}

		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{

		ptr_db   = db[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lamt = lamt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

		ll = 0;
		for(; ll<nb-3; ll+=4)
			{
			v_db0   = _mm256_load_pd( &ptr_db[ll] );
			v_db1   = _mm256_load_pd( &ptr_db[pnb+ll] );
			v_db1   = _mm256_xor_pd( v_db1, v_sign );
			v_dux   = _mm256_load_pd( &ptr_dux[ll] );
			v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
			v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
			v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
			_mm256_store_pd( &ptr_dt[ll], v_dt0 );
			_mm256_store_pd( &ptr_dt[pnb+ll], v_dt1 );

			v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
			v_lamt1 = _mm256_load_pd( &ptr_lamt[pnb+ll] );
			v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
			v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
			_mm256_store_pd( &ptr_dlam[ll], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[pnb+ll], v_dlam1 );

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
			t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
			t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_t      = _mm256_xor_ps( t_t, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp1   = _mm256_div_ps( t_t, t_dt );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
			t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

			}
		if(ll<nb)
			{

			ll_left = nb - ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_db0   = _mm256_load_pd( &ptr_db[ll] );
			v_db1   = _mm256_load_pd( &ptr_db[pnb+ll] );
			v_db1   = _mm256_xor_pd( v_db1, v_sign );
			v_dux   = _mm256_load_pd( &ptr_dux[ll] );
			v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
			v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
			v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
			_mm256_maskstore_pd( &ptr_dt[ll], i_mask, v_dt0 );
			_mm256_maskstore_pd( &ptr_dt[pnb+ll], i_mask, v_dt1 );

			v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
			v_lamt1 = _mm256_load_pd( &ptr_lamt[pnb+ll] );
			v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
			v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
			_mm256_maskstore_pd( &ptr_dlam[ll], i_mask, v_dlam0 );
			_mm256_maskstore_pd( &ptr_dlam[pnb+ll], i_mask, v_dlam1 );

			if(ll<nb-2) // 3 left
				{

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
				t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
				t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_t      = _mm256_xor_ps( t_t, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp1   = _mm256_div_ps( t_t, t_dt );
				t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
				t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
				t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

				}
			else // 1 or 2 left
				{

				s_mask   = _mm256_cvtpd_ps( v_mask );
				s_mask   = _mm_shuffle_ps( s_mask, s_mask, 0x44 );
				t_mask1  = _mm256_permute2f128_ps( _mm256_castps128_ps256( s_mask ), _mm256_castps128_ps256( s_mask ), 0x20 );
				t_mask1  = _mm256_cmp_ps( t_mask1, t_zeros, 0x01 );

				v_dt0    = _mm256_permute2f128_pd( v_dt0, v_dt1, 0x20 );
				v_t0     = _mm256_permute2f128_pd( v_t0, v_t1, 0x20 );
				v_dlam0  = _mm256_permute2f128_pd( v_dlam0, v_dlam1, 0x20 );
				v_lam0   = _mm256_permute2f128_pd( v_lam0, v_lam1, 0x20 );

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
				t_mask0  = _mm256_and_ps( t_mask0, t_mask1 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

				}
			}

		}		

	// last stage
	jj = N;

	ptr_db   = db[jj];
	ptr_dux  = dux[jj];
	ptr_t    = t[jj];
	ptr_dt   = dt[jj];
	ptr_lamt = lamt[jj];
	ptr_lam  = lam[jj];
	ptr_dlam = dlam[jj];

	ll = nu;
	// special case: nu=1, nx=2 // TODO mask before and after instead !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	if(ll==nb-2 && ll%2==1)
		{
		u_db0   = _mm_loadu_pd( &ptr_db[ll] );
		u_db1   = _mm_loadu_pd( &ptr_db[pnb+ll] );
		u_db1   = _mm_xor_pd( u_db1, u_sign );
		u_dux   = _mm_loadu_pd( &ptr_dux[ll] );
		u_dt0   = _mm_sub_pd ( u_dux, u_db0 );
		u_dt1   = _mm_sub_pd ( u_db1, u_dux );
		u_t0    = _mm_loadu_pd( &ptr_t[ll] );
		u_t1    = _mm_loadu_pd( &ptr_t[pnb+ll] );
		u_dt0   = _mm_sub_pd( u_dt0, u_t0 );
		u_dt1   = _mm_sub_pd( u_dt1, u_t1 );
		_mm_storeu_pd( &ptr_dt[ll], u_dt0 );
		_mm_storeu_pd( &ptr_dt[pnb+ll], u_dt1 );
		v_dt0   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dt0 ), _mm256_castpd128_pd256( u_dt1 ), 0x20 );
		v_t0    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_t0 ), _mm256_castpd128_pd256( u_t1 ), 0x20 );

		u_lamt0 = _mm_loadu_pd( &ptr_lamt[ll] );
		u_lamt1 = _mm_loadu_pd( &ptr_lamt[pnb+ll] );
		u_temp0 = _mm_mul_pd( u_lamt0, u_dt0 );
		u_temp1 = _mm_mul_pd( u_lamt1, u_dt1 );
		u_dlam0 = _mm_loadu_pd( &ptr_dlam[ll] );
		u_dlam1 = _mm_loadu_pd( &ptr_dlam[pnb+ll] );
		u_lam0  = _mm_loadu_pd( &ptr_lam[ll] );
		u_lam1  = _mm_loadu_pd( &ptr_lam[pnb+ll] );
		u_dlam0 = _mm_sub_pd( u_dlam0, u_lam0 );
		u_dlam1 = _mm_sub_pd( u_dlam1, u_lam1 );
		u_dlam0 = _mm_sub_pd( u_dlam0, u_temp0 );
		u_dlam1 = _mm_sub_pd( u_dlam1, u_temp1 );
		_mm_storeu_pd( &ptr_dlam[ll], u_dlam0 );
		_mm_storeu_pd( &ptr_dlam[pnb+ll], u_dlam1 );
		v_dlam0   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam0 ), _mm256_castpd128_pd256( u_dlam1 ), 0x20 );
		v_lam0    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam0 ), _mm256_castpd128_pd256( u_lam1 ), 0x20 );

		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

		ll+=2; // delete when in for
		}



	if(ll<nb && ll%2==1)
	//for(; ll<((nu+bs-1)/bs)*bs; ll++)
		{
		u_db0   = _mm_load_sd( &ptr_db[ll] );
		u_db1   = _mm_load_sd( &ptr_db[pnb+ll] );
		u_db1   = _mm_xor_pd( u_db1, u_sign );
		u_dux   = _mm_load_sd( &ptr_dux[ll] );
		u_dt0   = _mm_sub_sd ( u_dux, u_db0 );
		u_dt1   = _mm_sub_sd ( u_db1, u_dux );
		u_t0    = _mm_load_sd( &ptr_t[ll] );
		u_t1    = _mm_load_sd( &ptr_t[pnb+ll] );
		u_dt0   = _mm_sub_sd( u_dt0, u_t0 );
		u_dt1   = _mm_sub_sd( u_dt1, u_t1 );
		_mm_store_sd( &ptr_dt[ll], u_dt0 );
		_mm_store_sd( &ptr_dt[pnb+ll], u_dt1 );
		u_t0    = _mm_shuffle_pd( u_t0, u_t1, 0x0 );
		u_dt0   = _mm_shuffle_pd( u_dt0, u_dt1, 0x0 );

		u_lamt0 = _mm_load_sd( &ptr_lamt[ll] );
		u_lamt1 = _mm_load_sd( &ptr_lamt[pnb+ll] );
		u_temp0 = _mm_mul_sd( u_lamt0, u_dt0 );
		u_temp1 = _mm_mul_sd( u_lamt1, u_dt1 );
		u_dlam0 = _mm_load_sd( &ptr_dlam[ll] );
		u_dlam1 = _mm_load_sd( &ptr_dlam[pnb+ll] );
		u_lam0  = _mm_load_sd( &ptr_lam[ll] );
		u_lam1  = _mm_load_sd( &ptr_lam[pnb+ll] );
		u_dlam0 = _mm_sub_sd( u_dlam0, u_lam0 );
		u_dlam1 = _mm_sub_sd( u_dlam1, u_lam1 );
		u_dlam0 = _mm_sub_sd( u_dlam0, u_temp0 );
		u_dlam1 = _mm_sub_sd( u_dlam1, u_temp1 );
		_mm_store_sd( &ptr_dlam[ll], u_dlam0 );
		_mm_store_sd( &ptr_dlam[pnb+ll], u_dlam1 );
		u_lam0  = _mm_shuffle_pd( u_lam0, u_lam1, 0x0 );
		u_dlam0 = _mm_shuffle_pd( u_dlam0, u_dlam1, 0x0 );

		v_dlam0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam0 ), _mm256_castpd128_pd256( u_dt0 ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam0 );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam0  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam0 ), _mm256_castpd128_pd256( u_t0 ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam0 );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha0 = _mm_min_ps( s_alpha0, s_tmp0 );

		ll++; // delete when in for
		}
	if(ll<nb-1 && ll%4==2)
	//for(; ll<((nu+bs-1)/bs)*bs; ll++)
		{
		u_db0   = _mm_load_pd( &ptr_db[ll] );
		u_db1   = _mm_load_pd( &ptr_db[pnb+ll] );
		u_db1   = _mm_xor_pd( u_db1, u_sign );
		u_dux   = _mm_load_pd( &ptr_dux[ll] );
		u_dt0   = _mm_sub_pd ( u_dux, u_db0 );
		u_dt1   = _mm_sub_pd ( u_db1, u_dux );
		u_t0    = _mm_load_pd( &ptr_t[ll] );
		u_t1    = _mm_load_pd( &ptr_t[pnb+ll] );
		u_dt0   = _mm_sub_pd( u_dt0, u_t0 );
		u_dt1   = _mm_sub_pd( u_dt1, u_t1 );
		_mm_store_pd( &ptr_dt[ll], u_dt0 );
		_mm_store_pd( &ptr_dt[pnb+ll], u_dt1 );
		v_dt0   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dt0 ), _mm256_castpd128_pd256( u_dt1 ), 0x20 );
		v_t0    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_t0 ), _mm256_castpd128_pd256( u_t1 ), 0x20 );

		u_lamt0 = _mm_load_pd( &ptr_lamt[ll] );
		u_lamt1 = _mm_load_pd( &ptr_lamt[pnb+ll] );
		u_temp0 = _mm_mul_pd( u_lamt0, u_dt0 );
		u_temp1 = _mm_mul_pd( u_lamt1, u_dt1 );
		u_dlam0 = _mm_load_pd( &ptr_dlam[ll] );
		u_dlam1 = _mm_load_pd( &ptr_dlam[pnb+ll] );
		u_lam0  = _mm_load_pd( &ptr_lam[ll] );
		u_lam1  = _mm_load_pd( &ptr_lam[pnb+ll] );
		u_dlam0 = _mm_sub_pd( u_dlam0, u_lam0 );
		u_dlam1 = _mm_sub_pd( u_dlam1, u_lam1 );
		u_dlam0 = _mm_sub_pd( u_dlam0, u_temp0 );
		u_dlam1 = _mm_sub_pd( u_dlam1, u_temp1 );
		_mm_store_pd( &ptr_dlam[ll], u_dlam0 );
		_mm_store_pd( &ptr_dlam[pnb+ll], u_dlam1 );
		v_dlam0   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam0 ), _mm256_castpd128_pd256( u_dlam1 ), 0x20 );
		v_lam0    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam0 ), _mm256_castpd128_pd256( u_lam1 ), 0x20 );

		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

		ll+=2; // delete when in for
		}
		
	for(; ll<nb-3; ll+=4)
		{
		v_db0   = _mm256_load_pd( &ptr_db[ll] );
		v_db1   = _mm256_load_pd( &ptr_db[pnb+ll] );
		v_db1   = _mm256_xor_pd( v_db1, v_sign );
		v_dux   = _mm256_load_pd( &ptr_dux[ll] );
		v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
		v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
		v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
		_mm256_store_pd( &ptr_dt[ll], v_dt0 );
		_mm256_store_pd( &ptr_dt[pnb+ll], v_dt1 );

		v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
		v_lamt1 = _mm256_load_pd( &ptr_lamt[pnb+ll] );
		v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
		v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
		v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
		v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
		v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
		_mm256_store_pd( &ptr_dlam[ll], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[pnb+ll], v_dlam1 );

		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
		t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
		t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_t      = _mm256_xor_ps( t_t, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp1   = _mm256_div_ps( t_t, t_dt );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
		t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

		}
	if(ll<nb)
		{

		ll_left = nb - ll;
		v_left  = _mm256_broadcast_sd( &ll_left );
		v_mask  = _mm256_loadu_pd( d_mask );
		v_mask  = _mm256_sub_pd( v_mask, v_left );
		i_mask  = _mm256_castpd_si256( v_mask );

		v_db0   = _mm256_load_pd( &ptr_db[ll] );
		v_db1   = _mm256_load_pd( &ptr_db[pnb+ll] );
		v_db1   = _mm256_xor_pd( v_db1, v_sign );
		v_dux   = _mm256_load_pd( &ptr_dux[ll] );
		v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
		v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
		v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
		_mm256_maskstore_pd( &ptr_dt[ll], i_mask, v_dt0 );
		_mm256_maskstore_pd( &ptr_dt[pnb+ll], i_mask, v_dt1 );

		v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
		v_lamt1 = _mm256_load_pd( &ptr_lamt[pnb+ll] );
		v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
		v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
		v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
		v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
		v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
		_mm256_maskstore_pd( &ptr_dlam[ll], i_mask, v_dlam0 );
		_mm256_maskstore_pd( &ptr_dlam[pnb+ll], i_mask, v_dlam1 );

		if(ll<nbu-2) // 3 left
			{

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
			t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
			t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_t      = _mm256_xor_ps( t_t, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp1   = _mm256_div_ps( t_t, t_dt );
			t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
			t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
			t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

			}
		else // 1 or 2 left
			{

			s_mask   = _mm256_cvtpd_ps( v_mask );
			s_mask   = _mm_shuffle_ps( s_mask, s_mask, 0x44 );
			t_mask1  = _mm256_permute2f128_ps( _mm256_castps128_ps256( s_mask ), _mm256_castps128_ps256( s_mask ), 0x20 );
			t_mask1  = _mm256_cmp_ps( t_mask1, t_zeros, 0x01 );

			v_dt0    = _mm256_permute2f128_pd( v_dt0, v_dt1, 0x20 );
			v_t0     = _mm256_permute2f128_pd( v_t0, v_t1, 0x20 );
			v_dlam0  = _mm256_permute2f128_pd( v_dlam0, v_dlam1, 0x20 );
			v_lam0   = _mm256_permute2f128_pd( v_lam0, v_lam1, 0x20 );

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
			t_mask0  = _mm256_and_ps( t_mask0, t_mask1 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

			}

		}
	
	// general constraints
	if(ng>0)
		{

		for(jj=0; jj<N; jj++)
			{
		
			ptr_db   = db[jj];
			ptr_dux  = dux[jj];
			ptr_t    = t[jj];
			ptr_dt   = dt[jj];
			ptr_lamt = lamt[jj];
			ptr_lam  = lam[jj];
			ptr_dlam = dlam[jj];

			dgemv_t_lib(nx+nu, ng, pDCt[jj], cng, ptr_dux, 0, ptr_dt+2*pnb, ptr_dt+2*pnb);

			for(ll=2*pnb; ll<2*pnb+ng-3; ll+=4)
				{
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_xor_pd( v_dt0, v_sign );
				v_db0   = _mm256_load_pd( &ptr_db[ll] );
				v_db1   = _mm256_load_pd( &ptr_db[png+ll] );
				v_dt0   = _mm256_sub_pd ( v_dt0, v_db0 );
				v_dt1   = _mm256_sub_pd ( v_dt1, v_db1 );
				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
				v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
				v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
				_mm256_store_pd( &ptr_dt[ll], v_dt0 );
				_mm256_store_pd( &ptr_dt[png+ll], v_dt1 );

				v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
				v_lamt1 = _mm256_load_pd( &ptr_lamt[png+ll] );
				v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
				v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
				_mm256_store_pd( &ptr_dlam[ll], v_dlam0 );
				_mm256_store_pd( &ptr_dlam[png+ll], v_dlam1 );

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
				t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
				t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_t      = _mm256_xor_ps( t_t, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp1   = _mm256_div_ps( t_t, t_dt );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
				t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

				}
			if(ll<2*pnb+ng)
				{

				ll_left = 2*pnb + ng - ll;
				v_left  = _mm256_broadcast_sd( &ll_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				v_mask  = _mm256_sub_pd( v_mask, v_left );
				i_mask  = _mm256_castpd_si256( v_mask );

				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_xor_pd( v_dt0, v_sign );
				v_db0   = _mm256_load_pd( &ptr_db[ll] );
				v_db1   = _mm256_load_pd( &ptr_db[png+ll] );
				v_dt0   = _mm256_sub_pd ( v_dt0, v_db0 );
				v_dt1   = _mm256_sub_pd ( v_dt1, v_db1 );
				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
				v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
				v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
				_mm256_maskstore_pd( &ptr_dt[ll], i_mask, v_dt0 );
				_mm256_maskstore_pd( &ptr_dt[png+ll], i_mask, v_dt1 );

				v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
				v_lamt1 = _mm256_load_pd( &ptr_lamt[png+ll] );
				v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
				v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
				_mm256_maskstore_pd( &ptr_dlam[ll], i_mask, v_dlam0 );
				_mm256_maskstore_pd( &ptr_dlam[png+ll], i_mask, v_dlam1 );

				if(ll<2*pnb+ng-2) // 3 left
					{

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
					t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
					t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_t      = _mm256_xor_ps( t_t, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp1   = _mm256_div_ps( t_t, t_dt );
					t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
					t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
					t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

					}
				else // 1 or 2 left
					{

					s_mask   = _mm256_cvtpd_ps( v_mask );
					s_mask   = _mm_shuffle_ps( s_mask, s_mask, 0x44 );
					t_mask1  = _mm256_permute2f128_ps( _mm256_castps128_ps256( s_mask ), _mm256_castps128_ps256( s_mask ), 0x20 );
					t_mask1  = _mm256_cmp_ps( t_mask1, t_zeros, 0x01 );

					v_dt0    = _mm256_permute2f128_pd( v_dt0, v_dt1, 0x20 );
					v_t0     = _mm256_permute2f128_pd( v_t0, v_t1, 0x20 );
					v_dlam0  = _mm256_permute2f128_pd( v_dlam0, v_dlam1, 0x20 );
					v_lam0   = _mm256_permute2f128_pd( v_lam0, v_lam1, 0x20 );

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
					t_mask0  = _mm256_and_ps( t_mask0, t_mask1 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

					}
				}
			}
		}
	if(ngN>0)
		{

		ptr_db   = db[N];
		ptr_dux  = dux[N];
		ptr_t    = t[N];
		ptr_dt   = dt[N];
		ptr_lamt = lamt[N];
		ptr_lam  = lam[N];
		ptr_dlam = dlam[N];

		dgemv_t_lib(nx+nu, ngN, pDCt[N], cngN, ptr_dux, 0, ptr_dt+2*pnb, ptr_dt+2*pnb);

		for(ll=2*pnb; ll<2*pnb+ngN-3; ll+=4)
			{
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_xor_pd( v_dt0, v_sign );
			v_db0   = _mm256_load_pd( &ptr_db[ll] );
			v_db1   = _mm256_load_pd( &ptr_db[pngN+ll] );
			v_dt0   = _mm256_sub_pd ( v_dt0, v_db0 );
			v_dt1   = _mm256_sub_pd ( v_dt1, v_db1 );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pngN+ll] );
			v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
			v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
			_mm256_store_pd( &ptr_dt[ll], v_dt0 );
			_mm256_store_pd( &ptr_dt[pngN+ll], v_dt1 );

			v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
			v_lamt1 = _mm256_load_pd( &ptr_lamt[pngN+ll] );
			v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
			v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pngN+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pngN+ll] );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
			_mm256_store_pd( &ptr_dlam[ll], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[pngN+ll], v_dlam1 );

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
			t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
			t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_t      = _mm256_xor_ps( t_t, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp1   = _mm256_div_ps( t_t, t_dt );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
			t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

			}
		if(ll<2*pnb+ngN)
			{

			ll_left = 2*pnb + ngN - ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_xor_pd( v_dt0, v_sign );
			v_db0   = _mm256_load_pd( &ptr_db[ll] );
			v_db1   = _mm256_load_pd( &ptr_db[pngN+ll] );
			v_dt0   = _mm256_sub_pd ( v_dt0, v_db0 );
			v_dt1   = _mm256_sub_pd ( v_dt1, v_db1 );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pngN+ll] );
			v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
			v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
			_mm256_maskstore_pd( &ptr_dt[ll], i_mask, v_dt0 );
			_mm256_maskstore_pd( &ptr_dt[pngN+ll], i_mask, v_dt1 );

			v_lamt0 = _mm256_load_pd( &ptr_lamt[ll] );
			v_lamt1 = _mm256_load_pd( &ptr_lamt[pngN+ll] );
			v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
			v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pngN+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pngN+ll] );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
			v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
			v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
			_mm256_maskstore_pd( &ptr_dlam[ll], i_mask, v_dlam0 );
			_mm256_maskstore_pd( &ptr_dlam[pngN+ll], i_mask, v_dlam1 );

			if(ll<2*pnb+ngN-2) // 3 left
				{

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
				t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
				t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_t      = _mm256_xor_ps( t_t, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp1   = _mm256_div_ps( t_t, t_dt );
				t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
				t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
				t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

				}
			else // 1 or 2 left
				{

				s_mask   = _mm256_cvtpd_ps( v_mask );
				s_mask   = _mm_shuffle_ps( s_mask, s_mask, 0x44 );
				t_mask1  = _mm256_permute2f128_ps( _mm256_castps128_ps256( s_mask ), _mm256_castps128_ps256( s_mask ), 0x20 );
				t_mask1  = _mm256_cmp_ps( t_mask1, t_zeros, 0x01 );

				v_dt0    = _mm256_permute2f128_pd( v_dt0, v_dt1, 0x20 );
				v_t0     = _mm256_permute2f128_pd( v_t0, v_t1, 0x20 );
				v_dlam0  = _mm256_permute2f128_pd( v_dlam0, v_dlam1, 0x20 );
				v_lam0   = _mm256_permute2f128_pd( v_lam0, v_lam1, 0x20 );

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
				t_mask0  = _mm256_and_ps( t_mask0, t_mask1 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

				}
			}

		}



	// reduce alpha
	t_alpha0 = _mm256_min_ps( t_alpha0, t_alpha1 );
	s_alpha0 = _mm256_extractf128_ps( t_alpha0, 0x1 );
//	s_alpha1 = _mm256_extractf128_ps( t_alpha0, 0x1 );
//	s_alpha0  = _mm_min_ps( s_alpha0 , s_alpha1 );
	s_alpha1 = _mm256_castps256_ps128( t_alpha0 );
	s_alpha0 = _mm_min_ps( s_alpha0, s_alpha1 );
	
	v_alpha = _mm256_cvtps_pd( s_alpha0 );
	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );
	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );
	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
	_mm_store_sd( &alpha, u_alpha );

	
	ptr_alpha[0] = alpha;

	return;
	
	}


void d_compute_alpha_soft_mpc(int N, int nx, int nu, int nh, int ns, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db, double **Zl, double **zl)
	{

	int nb = nh + ns;
	
	int nhu = nu<nh ? nu : nh ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	__m256
		t_sign, t_ones, t_zeros,
		t_mask0, t_mask1,
		t_lam, t_dlam, t_t, t_dt,
		t_lam_z, t_dlam_z,
		t_tmp0, t_tmp1,
		t_alpha0, t_alpha1;
		
	__m128
		s_sign, s_ones, s_zeros,
		s_mask0, s_mask1,
		s_lam, s_dlam, s_t, s_dt,
		s_lam_z, s_dlam_z, s_t_z, s_dt_z,
		s_tmp0, s_tmp1,
		s_alpha0, s_alpha1;
	
	__m256d
		v_sign, v_temp, v_tmp0, v_tmp1,
		v_dt, v_dux, v_db, v_dlam, v_lamt, v_t, v_alpha, v_lam,
		v_dt_z, v_dlam_z, v_lamt_z, v_t_z, v_lam_z;
	
	__m128d
		u_sign, u_temp, u_tmp0, u_tmp1,
		u_dux, u_db, u_alpha, 
		u_dt, u_dlam, u_lamt, u_t, u_lam,
		u_dt_z, u_dlam_z, u_lamt_z, u_t_z, u_lam_z;
	
	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );
	u_sign = _mm_loaddup_pd( (double *) &long_sign );

	int int_sign = 0x80000000;
	s_sign = _mm_broadcast_ss( (float *) &int_sign );
	t_sign = _mm256_broadcast_ss( (float *) &int_sign );
	
	s_ones  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_zeros = _mm_setzero_ps( );

	t_ones  = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_zeros = _mm256_setzero_ps( );

	s_alpha0 = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_alpha1 = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );

	t_alpha0 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_alpha1 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	
	double alpha;

	int jj, ll;


	// first stage
	jj = 0;

	ll = 0;
	// hard input constraints
	for(; ll<nhu-1; ll+=2)
		{

		v_db    = _mm256_load_pd( &db[jj][2*ll] );
		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_t     = _mm256_load_pd( &t[jj][2*ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[jj][2*ll], v_dt );

		v_lamt  = _mm256_load_pd( &lamt[jj][2*ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dt );
		v_dlam  = _mm256_load_pd( &dlam[jj][2*ll] );
		v_lam   = _mm256_load_pd( &lam[jj][2*ll] );
		v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
		_mm256_store_pd( &dlam[jj][2*ll], v_dlam );

		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

		}

	for(; ll<nhu; ll++)
		{

		u_db     = _mm_load_pd( &db[jj][2*ll] );
		u_dux    = _mm_loaddup_pd( &dux[jj][ll+0] );
		u_dt     = _mm_addsub_pd( u_db, u_dux );
		u_dt     = _mm_xor_pd( u_dt, u_sign );
		u_t      = _mm_load_pd( &t[jj][2*ll] );
		u_dt     = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[jj][2*ll], u_dt );

		u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
		u_temp   = _mm_mul_pd( u_lamt, u_dt );
		u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
		u_lam    = _mm_load_pd( &lam[jj][2*ll] );
		u_dlam   = _mm_sub_pd( u_dlam, u_lam );
		u_dlam   = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[jj][2*ll], u_dlam );

		v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam   = _mm256_cvtpd_ps( v_dlam );
		s_mask0  = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam    = _mm256_cvtpd_ps( v_lam );
		s_lam    = _mm_xor_ps( s_lam, s_sign );
		s_tmp0   = _mm_div_ps( s_lam, s_dlam );
		s_tmp0   = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha0 = _mm_min_ps( s_alpha0, s_tmp0 );

		}
	
	// middle stages
	for(jj=1; jj<N; jj++)
		{

		ll = 0;
		// hard input and state constraints
		for(; ll<nhu-1; ll+=2)
			{

			v_db    = _mm256_load_pd( &db[jj][2*ll] );
			v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+0] ) );
			v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+1] ) );
			v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
			v_dt    = _mm256_addsub_pd( v_db, v_dux );
			v_dt    = _mm256_xor_pd( v_dt, v_sign );
			v_t     = _mm256_load_pd( &t[jj][2*ll] );
			v_dt    = _mm256_sub_pd( v_dt, v_t );
			_mm256_store_pd( &dt[jj][2*ll], v_dt );

			v_lamt  = _mm256_load_pd( &lamt[jj][2*ll] );
			v_temp  = _mm256_mul_pd( v_lamt, v_dt );
			v_dlam  = _mm256_load_pd( &dlam[jj][2*ll] );
			v_lam   = _mm256_load_pd( &lam[jj][2*ll] );
			v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
			v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
			_mm256_store_pd( &dlam[jj][2*ll], v_dlam );

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

			}

		for(; ll<nhu; ll++)
			{

			u_db     = _mm_load_pd( &db[jj][2*ll] );
			u_dux    = _mm_loaddup_pd( &dux[jj][ll+0] );
			u_dt     = _mm_addsub_pd( u_db, u_dux );
			u_dt     = _mm_xor_pd( u_dt, u_sign );
			u_t      = _mm_load_pd( &t[jj][2*ll] );
			u_dt     = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][2*ll], u_dt );

			u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
			u_temp   = _mm_mul_pd( u_lamt, u_dt );
			u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
			u_lam    = _mm_load_pd( &lam[jj][2*ll] );
			u_dlam   = _mm_sub_pd( u_dlam, u_lam );
			u_dlam   = _mm_sub_pd( u_dlam, u_temp );
			_mm_store_pd( &dlam[jj][2*ll], u_dlam );

			v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
			s_dlam   = _mm256_cvtpd_ps( v_dlam );
			s_mask0  = _mm_cmplt_ps( s_dlam, s_zeros );
			v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
			s_lam    = _mm256_cvtpd_ps( v_lam );
			s_lam    = _mm_xor_ps( s_lam, s_sign );
			s_tmp0   = _mm_div_ps( s_lam, s_dlam );
			s_tmp0   = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
			s_alpha0 = _mm_min_ps( s_alpha0, s_tmp0 );

			}

		// soft state constraints
		if(ll<nb && ll%2==1)
			{

			u_dux   = _mm_loaddup_pd( &dux[jj][ll+0] );
			u_lamt  = _mm_load_pd( &lamt[jj][2*ll] );
			u_temp  = _mm_mul_pd( u_lamt, u_dux );
			u_dt_z  = _mm_load_pd( &zl[jj][2*ll] );
			u_dt_z  = _mm_addsub_pd( u_dt_z, u_temp );
			u_temp  = _mm_load_pd( &Zl[jj][2*ll] );
			u_dt_z  = _mm_mul_pd( u_dt_z, u_temp );
			u_db    = _mm_load_pd( &db[jj][2*ll] );
			u_dt    = _mm_addsub_pd( u_db, u_dux );
			u_dt    = _mm_xor_pd( u_dt, u_sign );
			u_dt    = _mm_add_pd( u_dt, u_dt_z );
			u_t     = _mm_load_pd( &t[jj][2*ll] );
			u_dt    = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][2*ll], u_dt );
			u_t_z   = _mm_load_pd( &t[jj][pnb+2*ll] );
			u_dt_z  = _mm_sub_pd( u_dt_z, u_t_z );
			_mm_store_pd( &dt[jj][pnb+2*ll], u_dt_z );

			//u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
			u_lamt_z = _mm_load_pd( &lamt[jj][pnb+2*ll] );
			u_tmp0   = _mm_mul_pd( u_lamt, u_dt );
			u_tmp1   = _mm_mul_pd( u_lamt_z, u_dt_z );
			u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
			u_dlam_z = _mm_load_pd( &dlam[jj][pnb+2*ll] );
			u_lam    = _mm_load_pd( &lam[jj][2*ll] );
			u_lam_z  = _mm_load_pd( &lam[jj][pnb+2*ll] );
			u_dlam   = _mm_sub_pd( u_dlam, u_lam );
			u_dlam_z = _mm_sub_pd( u_dlam_z, u_lam_z );
			u_dlam   = _mm_sub_pd( u_dlam, u_tmp0 );
			u_dlam_z = _mm_sub_pd( u_dlam_z, u_tmp1 );
			_mm_store_pd( &dlam[jj][2*ll], u_dlam );
			_mm_store_pd( &dlam[jj][pnb+2*ll], u_dlam_z );

			v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
			v_dlam_z = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam_z ), _mm256_castpd128_pd256( u_dt_z ), 0x20 );
			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
			v_lam_z  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam_z ), _mm256_castpd128_pd256( u_t_z ), 0x20 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

			ll++;

			}

		for(; ll<nb-1; ll+=2)
			{

			v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+0] ) );
			v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+1] ) );
			v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
			v_lamt  = _mm256_load_pd( &lamt[jj][2*ll] );
			v_temp  = _mm256_mul_pd( v_lamt, v_dux );
			v_dt_z  = _mm256_load_pd( &zl[jj][2*ll] );
			v_dt_z  = _mm256_addsub_pd( v_dt_z, v_temp );
			v_temp  = _mm256_load_pd( &Zl[jj][2*ll] );
			v_dt_z  = _mm256_mul_pd( v_dt_z, v_temp );
			v_db    = _mm256_load_pd( &db[jj][2*ll] );
			v_dt    = _mm256_addsub_pd( v_db, v_dux );
			v_dt    = _mm256_xor_pd( v_dt, v_sign );
			v_dt    = _mm256_add_pd( v_dt, v_dt_z );
			v_t     = _mm256_load_pd( &t[jj][2*ll] );
			v_dt    = _mm256_sub_pd( v_dt, v_t );
			_mm256_store_pd( &dt[jj][2*ll], v_dt );
			v_t_z   = _mm256_load_pd( &t[jj][pnb+2*ll] );
			v_dt_z  = _mm256_sub_pd( v_dt_z, v_t_z );
			_mm256_store_pd( &dt[jj][pnb+2*ll], v_dt_z );

			//v_lamt   = _mm256_load_pd( &lamt[jj][2*ll] );
			v_lamt_z = _mm256_load_pd( &lamt[jj][pnb+2*ll] );
			v_tmp0   = _mm256_mul_pd( v_lamt, v_dt );
			v_tmp1   = _mm256_mul_pd( v_lamt_z, v_dt_z );
			v_dlam   = _mm256_load_pd( &dlam[jj][2*ll] );
			v_dlam_z = _mm256_load_pd( &dlam[jj][pnb+2*ll] );
			v_lam    = _mm256_load_pd( &lam[jj][2*ll] );
			v_lam_z  = _mm256_load_pd( &lam[jj][pnb+2*ll] );
			v_dlam   = _mm256_sub_pd( v_dlam, v_lam );
			v_dlam_z = _mm256_sub_pd( v_dlam_z, v_lam_z );
			v_dlam   = _mm256_sub_pd( v_dlam, v_tmp0 );
			v_dlam_z = _mm256_sub_pd( v_dlam_z, v_tmp1 );
			_mm256_store_pd( &dlam[jj][2*ll], v_dlam );
			_mm256_store_pd( &dlam[jj][pnb+2*ll], v_dlam_z );

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt ) ), 0x20 );
			t_dlam_z = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt_z ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_mask1  = _mm256_cmp_ps( t_dlam_z, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t ) ), 0x20 );
			t_lam_z  = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t_z ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_lam_z  = _mm256_xor_ps( t_lam_z, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp1   = _mm256_div_ps( t_lam_z, t_dlam_z );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
			t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

			}

		for(; ll<nb; ll++)
			{

			u_dux   = _mm_loaddup_pd( &dux[jj][ll+0] );
			u_lamt  = _mm_load_pd( &lamt[jj][2*ll] );
			u_temp  = _mm_mul_pd( u_lamt, u_dux );
			u_dt_z  = _mm_load_pd( &zl[jj][2*ll] );
			u_dt_z  = _mm_addsub_pd( u_dt_z, u_temp );
			u_temp  = _mm_load_pd( &Zl[jj][2*ll] );
			u_dt_z  = _mm_mul_pd( u_dt_z, u_temp );
			u_db    = _mm_load_pd( &db[jj][2*ll] );
			u_dt    = _mm_addsub_pd( u_db, u_dux );
			u_dt    = _mm_xor_pd( u_dt, u_sign );
			u_dt    = _mm_add_pd( u_dt, u_dt_z );
			u_t     = _mm_load_pd( &t[jj][2*ll] );
			u_dt    = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][2*ll], u_dt );
			u_t_z   = _mm_load_pd( &t[jj][pnb+2*ll] );
			u_dt_z  = _mm_sub_pd( u_dt_z, u_t_z );
			_mm_store_pd( &dt[jj][pnb+2*ll], u_dt_z );

			//u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
			u_lamt_z = _mm_load_pd( &lamt[jj][pnb+2*ll] );
			u_tmp0   = _mm_mul_pd( u_lamt, u_dt );
			u_tmp1   = _mm_mul_pd( u_lamt_z, u_dt_z );
			u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
			u_dlam_z = _mm_load_pd( &dlam[jj][pnb+2*ll] );
			u_lam    = _mm_load_pd( &lam[jj][2*ll] );
			u_lam_z  = _mm_load_pd( &lam[jj][pnb+2*ll] );
			u_dlam   = _mm_sub_pd( u_dlam, u_lam );
			u_dlam_z = _mm_sub_pd( u_dlam_z, u_lam_z );
			u_dlam   = _mm_sub_pd( u_dlam, u_tmp0 );
			u_dlam_z = _mm_sub_pd( u_dlam_z, u_tmp1 );
			_mm_store_pd( &dlam[jj][2*ll], u_dlam );
			_mm_store_pd( &dlam[jj][pnb+2*ll], u_dlam_z );

			v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
			v_dlam_z = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam_z ), _mm256_castpd128_pd256( u_dt_z ), 0x20 );
			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
			v_lam_z  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam_z ), _mm256_castpd128_pd256( u_t_z ), 0x20 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

			}

		}		

	// last stage
	jj = N;

	ll = nu;
	// hard input constraints
	if(ll<nh && ll%2==1)
		{

		u_db    = _mm_load_pd( &db[N][2*ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[N][2*ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][2*ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[N][2*ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][2*ll] );
		u_lam   = _mm_load_pd( &lam[N][2*ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		_mm_store_pd( &dlam[N][2*ll], u_dlam );

		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha0 = _mm_min_ps( s_alpha0, s_tmp0 );

		ll++;

		}
	for(; ll<nh-1; ll+=2)
		{

		v_db    = _mm256_load_pd( &db[jj][2*ll] );
		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_t     = _mm256_load_pd( &t[jj][2*ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[jj][2*ll], v_dt );

		v_lamt  = _mm256_load_pd( &lamt[jj][2*ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dt );
		v_dlam  = _mm256_load_pd( &dlam[jj][2*ll] );
		v_lam   = _mm256_load_pd( &lam[jj][2*ll] );
		v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
		_mm256_store_pd( &dlam[jj][2*ll], v_dlam );

		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

		}
	for(; ll<nh; ll++)
		{

		u_db    = _mm_load_pd( &db[N][2*ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[N][2*ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][2*ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[N][2*ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][2*ll] );
		u_lam   = _mm_load_pd( &lam[N][2*ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[N][2*ll], u_dlam );

		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha0 = _mm_min_ps( s_alpha0, s_tmp0 );

		}

	// soft state constraints
	if(ll<nb && ll%2==1)
		{

		u_dux   = _mm_loaddup_pd( &dux[jj][ll+0] );
		u_lamt  = _mm_load_pd( &lamt[jj][2*ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dux );
		u_dt_z  = _mm_load_pd( &zl[jj][2*ll] );
		u_dt_z  = _mm_addsub_pd( u_dt_z, u_temp );
		u_temp  = _mm_load_pd( &Zl[jj][2*ll] );
		u_dt_z  = _mm_mul_pd( u_dt_z, u_temp );
		u_db    = _mm_load_pd( &db[jj][2*ll] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_dt    = _mm_add_pd( u_dt, u_dt_z );
		u_t     = _mm_load_pd( &t[jj][2*ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[jj][2*ll], u_dt );
		u_t_z   = _mm_load_pd( &t[jj][pnb+2*ll] );
		u_dt_z  = _mm_sub_pd( u_dt_z, u_t_z );
		_mm_store_pd( &dt[jj][pnb+2*ll], u_dt_z );

		//u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
		u_lamt_z = _mm_load_pd( &lamt[jj][pnb+2*ll] );
		u_tmp0   = _mm_mul_pd( u_lamt, u_dt );
		u_tmp1   = _mm_mul_pd( u_lamt_z, u_dt_z );
		u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
		u_dlam_z = _mm_load_pd( &dlam[jj][pnb+2*ll] );
		u_lam    = _mm_load_pd( &lam[jj][2*ll] );
		u_lam_z  = _mm_load_pd( &lam[jj][pnb+2*ll] );
		u_dlam   = _mm_sub_pd( u_dlam, u_lam );
		u_dlam_z = _mm_sub_pd( u_dlam_z, u_lam_z );
		u_dlam   = _mm_sub_pd( u_dlam, u_tmp0 );
		u_dlam_z = _mm_sub_pd( u_dlam_z, u_tmp1 );
		_mm_store_pd( &dlam[jj][2*ll], u_dlam );
		_mm_store_pd( &dlam[jj][pnb+2*ll], u_dlam_z );

		v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		v_dlam_z = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam_z ), _mm256_castpd128_pd256( u_dt_z ), 0x20 );
		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		v_lam_z  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam_z ), _mm256_castpd128_pd256( u_t_z ), 0x20 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

		ll++;

		}

	for(; ll<nb-1; ll+=2)
		{

		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_lamt  = _mm256_load_pd( &lamt[jj][2*ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dux );
		v_dt_z  = _mm256_load_pd( &zl[jj][2*ll] );
		v_dt_z  = _mm256_addsub_pd( v_dt_z, v_temp );
		v_temp  = _mm256_load_pd( &Zl[jj][2*ll] );
		v_dt_z  = _mm256_mul_pd( v_dt_z, v_temp );
		v_db    = _mm256_load_pd( &db[jj][2*ll] );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_dt    = _mm256_add_pd( v_dt, v_dt_z );
		v_t     = _mm256_load_pd( &t[jj][2*ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[jj][2*ll], v_dt );
		v_t_z   = _mm256_load_pd( &t[jj][pnb+2*ll] );
		v_dt_z  = _mm256_sub_pd( v_dt_z, v_t_z );
		_mm256_store_pd( &dt[jj][pnb+2*ll], v_dt_z );

		//v_lamt   = _mm256_load_pd( &lamt[jj][2*ll] );
		v_lamt_z = _mm256_load_pd( &lamt[jj][pnb+2*ll] );
		v_tmp0   = _mm256_mul_pd( v_lamt, v_dt );
		v_tmp1   = _mm256_mul_pd( v_lamt_z, v_dt_z );
		v_dlam   = _mm256_load_pd( &dlam[jj][2*ll] );
		v_dlam_z = _mm256_load_pd( &dlam[jj][pnb+2*ll] );
		v_lam    = _mm256_load_pd( &lam[jj][2*ll] );
		v_lam_z  = _mm256_load_pd( &lam[jj][pnb+2*ll] );
		v_dlam   = _mm256_sub_pd( v_dlam, v_lam );
		v_dlam_z = _mm256_sub_pd( v_dlam_z, v_lam_z );
		v_dlam   = _mm256_sub_pd( v_dlam, v_tmp0 );
		v_dlam_z = _mm256_sub_pd( v_dlam_z, v_tmp1 );
		_mm256_store_pd( &dlam[jj][2*ll], v_dlam );
		_mm256_store_pd( &dlam[jj][pnb+2*ll], v_dlam_z );

		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt ) ), 0x20 );
		t_dlam_z = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt_z ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		t_mask1  = _mm256_cmp_ps( t_dlam_z, t_zeros, 0x01 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t ) ), 0x20 );
		t_lam_z  = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t_z ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_lam_z  = _mm256_xor_ps( t_lam_z, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp1   = _mm256_div_ps( t_lam_z, t_dlam_z );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
		t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

		}

	for(; ll<nb; ll++)
		{

		u_dux   = _mm_loaddup_pd( &dux[jj][ll+0] );
		u_lamt  = _mm_load_pd( &lamt[jj][2*ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dux );
		u_dt_z  = _mm_load_pd( &zl[jj][2*ll] );
		u_dt_z  = _mm_addsub_pd( u_dt_z, u_temp );
		u_temp  = _mm_load_pd( &Zl[jj][2*ll] );
		u_dt_z  = _mm_mul_pd( u_dt_z, u_temp );
		u_db    = _mm_load_pd( &db[jj][2*ll] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_dt    = _mm_add_pd( u_dt, u_dt_z );
		u_t     = _mm_load_pd( &t[jj][2*ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[jj][2*ll], u_dt );
		u_t_z   = _mm_load_pd( &t[jj][pnb+2*ll] );
		u_dt_z  = _mm_sub_pd( u_dt_z, u_t_z );
		_mm_store_pd( &dt[jj][pnb+2*ll], u_dt_z );

		//u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
		u_lamt_z = _mm_load_pd( &lamt[jj][pnb+2*ll] );
		u_tmp0   = _mm_mul_pd( u_lamt, u_dt );
		u_tmp1   = _mm_mul_pd( u_lamt_z, u_dt_z );
		u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
		u_dlam_z = _mm_load_pd( &dlam[jj][pnb+2*ll] );
		u_lam    = _mm_load_pd( &lam[jj][2*ll] );
		u_lam_z  = _mm_load_pd( &lam[jj][pnb+2*ll] );
		u_dlam   = _mm_sub_pd( u_dlam, u_lam );
		u_dlam_z = _mm_sub_pd( u_dlam_z, u_lam_z );
		u_dlam   = _mm_sub_pd( u_dlam, u_tmp0 );
		u_dlam_z = _mm_sub_pd( u_dlam_z, u_tmp1 );
		_mm_store_pd( &dlam[jj][2*ll], u_dlam );
		_mm_store_pd( &dlam[jj][pnb+2*ll], u_dlam_z );

		v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		v_dlam_z = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam_z ), _mm256_castpd128_pd256( u_dt_z ), 0x20 );
		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		v_lam_z  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam_z ), _mm256_castpd128_pd256( u_t_z ), 0x20 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

		}
	
	// reduce alpha
	t_alpha0 = _mm256_min_ps( t_alpha0, t_alpha1 );
	s_alpha1 = _mm256_extractf128_ps( t_alpha0, 0x1 );
	s_alpha0  = _mm_min_ps( s_alpha0 , s_alpha1 );
	s_alpha1 = _mm256_castps256_ps128( t_alpha0 );
	s_alpha0 = _mm_min_ps( s_alpha0, s_alpha1 );
	
	v_alpha = _mm256_cvtps_pd( s_alpha0 );
	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );
	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );
	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
	_mm_store_sd( &alpha, u_alpha );

	ptr_alpha[0] = alpha;

	return;
	
	}



void d_compute_alpha_diag_mpc(int N, int *nx, int *nu, int *nb, int **idxb, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	double alpha = ptr_alpha[0];
	
/*	int kna = ((k1+bs-1)/bs)*bs;*/

	double
		*ptr_db, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lamt, *ptr_lam, *ptr_dlam;
	
	int jj, ll;

	int pnb;

	for(jj=0; jj<=N; jj++)
		{

		pnb  = bs*((nb[jj]+bs-1)/bs); // simd aligned number of box and soft constraints

		ptr_db   = db[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lamt = lamt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

		ll = 0;
		for(; ll<nb[jj]; ll++)
			{

			ptr_dt[ll+0]   =   ptr_dux[idxb[jj][ll]] - ptr_db[ll+0]   - ptr_t[ll+0];
			ptr_dt[ll+pnb] = - ptr_dux[idxb[jj][ll]] - ptr_db[ll+pnb] - ptr_t[ll+pnb];
			ptr_dlam[ll+0]   -= ptr_lamt[ll+0]   * ptr_dt[ll+0]   + ptr_lam[ll+0];
			ptr_dlam[ll+pnb] -= ptr_lamt[ll+pnb] * ptr_dt[ll+pnb] + ptr_lam[ll+pnb];
			if( -alpha*ptr_dlam[ll+0]>ptr_lam[ll+0] )
				{
				alpha = - ptr_lam[ll+0] / ptr_dlam[ll+0];
				}
			if( -alpha*ptr_dlam[ll+pnb]>ptr_lam[ll+pnb] )
				{
				alpha = - ptr_lam[ll+pnb] / ptr_dlam[ll+pnb];
				}
			if( -alpha*ptr_dt[ll+0]>ptr_t[ll+0] )
				{
				alpha = - ptr_t[ll+0] / ptr_dt[ll+0];
				}
			if( -alpha*ptr_dt[ll+pnb]>ptr_t[ll+pnb] )
				{
				alpha = - ptr_t[ll+pnb] / ptr_dt[ll+pnb];
				}

			}

		}		

	// store alpha
	ptr_alpha[0] = alpha;

	return;
	
	}



void d_update_var_hard_mpc_tv(int N, int *nx, int *nu, int *nb, int *ng, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	int nu0, nx0, nb0, pnb, ng0, png;

	int jj, ll, ll_bkp, ll_end;
	double ll_left;
	
	double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
	
	__m128d
		u_mu0, u_tmp;

	__m256d
		v_mask, v_left, v_zeros,
		v_alpha, v_ux, v_dux, v_pi, v_dpi, 
		v_t0, v_dt0, v_lam0, v_dlam0, v_mu0,
		v_t1, v_dt1, v_lam1, v_dlam1, v_mu1;
		
	__m256i
		i_mask;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_zeros = _mm256_setzero_pd();
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();

	double
		*ptr_pi, *ptr_dpi, *ptr_ux, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lam, *ptr_dlam;

	for(jj=0; jj<=N; jj++)
		{

		nx0 = nx[jj];
		nu0 = nu[jj];
		nb0 = nb[jj];
		pnb  = bs*((nb0+bs-1)/bs); // cache aligned number of box constraints
		ng0 = ng[jj];
		png  = bs*((ng0+bs-1)/bs); // cache aligned number of box constraints
		
		ptr_pi   = pi[jj];
		ptr_dpi  = dpi[jj];
		ptr_ux   = ux[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

		ll = 0;
		for(; ll<nx0-3; ll+=4)
			{
			v_pi  = _mm256_load_pd( &ptr_pi[ll] );
			v_dpi = _mm256_load_pd( &ptr_dpi[ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
#if defined(TARGET_X64_AVX2)
			v_pi  = _mm256_fmadd_pd( v_alpha, v_dpi, v_pi );
#else
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
#endif
			_mm256_store_pd( &ptr_pi[ll], v_pi );
			}
		if(ll<nx0)
			{
			ll_left = nx0-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_pi  = _mm256_load_pd( &ptr_pi[ll] );
			v_dpi = _mm256_load_pd( &ptr_dpi[ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
#if defined(TARGET_X64_AVX2)
			v_pi  = _mm256_fmadd_pd( v_alpha, v_dpi, v_pi );
#else
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
#endif
			_mm256_maskstore_pd( &ptr_pi[ll], i_mask, v_pi );
			}

		// update inputs & states
		// box constraints
		ll = 0;
		for(; ll<nb0-3; ll+=4)
			{
			v_ux    = _mm256_load_pd( &ptr_ux[ll] );
			v_dux   = _mm256_load_pd( &ptr_dux[ll] );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
			v_dux   = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_dux   = _mm256_mul_pd( v_alpha, v_dux );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_ux    = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_store_pd( &ptr_t[ll], v_t0 );
			_mm256_store_pd( &ptr_t[pnb+ll], v_t1 );
			_mm256_store_pd( &ptr_lam[ll], v_lam0 );
			_mm256_store_pd( &ptr_lam[pnb+ll], v_lam1 );
			_mm256_store_pd( &ptr_ux[ll], v_ux );
#if defined(TARGET_X64_AVX2)
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<nb0 && nb0==nx0+nu0)
			{
			ll_left = nb0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_ux    = _mm256_load_pd( &ptr_ux[ll] );
			v_dux   = _mm256_load_pd( &ptr_dux[ll] );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
			v_dux   = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_dux   = _mm256_mul_pd( v_alpha, v_dux );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_ux    = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[pnb+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[pnb+ll], i_mask, v_lam1 );
			_mm256_maskstore_pd( &ptr_ux[ll], i_mask, v_ux );
#if defined(TARGET_X64_AVX2)
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		else
			{
			// backup ll
			ll_bkp = ll;
			// clean up inputs & states
			for(; ll<nu0+nx0-3; ll+=4)
				{
				v_ux  = _mm256_load_pd( &ptr_ux[ll] );
				v_dux = _mm256_load_pd( &ptr_dux[ll] );
				v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
				v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
				v_dux = _mm256_mul_pd( v_alpha, v_dux );
				v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
				_mm256_store_pd( &ptr_ux[ll], v_ux );
				}
			if(ll<nu0+nx0)
				{
				ll_left = nu0+nx0-ll;
				v_left= _mm256_broadcast_sd( &ll_left );
				v_mask= _mm256_loadu_pd( d_mask );
				i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

				v_ux  = _mm256_load_pd( &ptr_ux[ll] );
				v_dux = _mm256_load_pd( &ptr_dux[ll] );
				v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
				v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
				v_dux = _mm256_mul_pd( v_alpha, v_dux );
				v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
				_mm256_maskstore_pd( &ptr_ux[ll], i_mask, v_ux );
				}
			// cleanup box constraints
			ll = ll_bkp;
			for(; ll<nb0-3; ll+=4)
				{
				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
				v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
				v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
				v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
				v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
				v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
				v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
				v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
				v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
				v_t0    = _mm256_add_pd( v_t0, v_dt0 );
				v_t1    = _mm256_add_pd( v_t1, v_dt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
				_mm256_store_pd( &ptr_t[ll], v_t0 );
				_mm256_store_pd( &ptr_t[pnb+ll], v_t1 );
				_mm256_store_pd( &ptr_lam[ll], v_lam0 );
				_mm256_store_pd( &ptr_lam[pnb+ll], v_lam1 );
#if defined(TARGET_X64_AVX2)
				v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
				v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
				v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
				v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
				v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
				v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
				}
			if(ll<nb0)
				{
				ll_left = nb0-ll;
				v_left  = _mm256_broadcast_sd( &ll_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				v_mask  = _mm256_sub_pd( v_mask, v_left );
				i_mask  = _mm256_castpd_si256( v_mask );

				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
				v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
				v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
				v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
				v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
				v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
				v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
				v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
				v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
				v_t0    = _mm256_add_pd( v_t0, v_dt0 );
				v_t1    = _mm256_add_pd( v_t1, v_dt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
				_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
				_mm256_maskstore_pd( &ptr_t[pnb+ll], i_mask, v_t1 );
				_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
				_mm256_maskstore_pd( &ptr_lam[pnb+ll], i_mask, v_lam1 );
#if defined(TARGET_X64_AVX2)
				v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
				v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
				v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
				v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
				v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
				v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
				v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
				v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
				v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
				v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
				}
			}

		// genreal constraints
		for(ll=2*pnb; ll<2*pnb+ng0-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[png+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_store_pd( &ptr_t[ll], v_t0 );
			_mm256_store_pd( &ptr_t[png+ll], v_t1 );
			_mm256_store_pd( &ptr_lam[ll], v_lam0 );
			_mm256_store_pd( &ptr_lam[png+ll], v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<2*pnb+ng0)
			{

			ll_left = 2*pnb+ng0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[png+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[png+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[png+ll], i_mask, v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif

			}

		}

	v_mu0 = _mm256_add_pd( v_mu0, v_mu1 );
	u_mu0 = _mm_add_pd( _mm256_castpd256_pd128( v_mu0 ), _mm256_extractf128_pd( v_mu0, 0x1 ) );
	u_mu0 = _mm_hadd_pd( u_mu0, u_mu0 );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu0 = _mm_mul_sd( u_mu0, u_tmp );
	_mm_store_sd( ptr_mu, u_mu0 );

	return;
	
	}


void d_update_var_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{
	
	const int nbu = nu<nb ? nu : nb ;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	//const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!
	const int pnb  = bs*((nb+bs-1)/bs); // simd aligned number of box and soft constraints
	const int png  = bs*((ng+bs-1)/bs); // simd aligned number of box and soft constraints
	const int pngN = bs*((ngN+bs-1)/bs); // simd aligned number of box and soft constraints at last stage

	int jj, ll, ll_bkp, ll_end;
	double ll_left;
	
	double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
	
	__m128d
		u_ux, u_dux, u_pi, u_dpi, u_tmp, 
		u_t0, u_dt0, u_lam0, u_dlam0, u_mu0,
		u_t1, u_dt1, u_lam1, u_dlam1, u_mu1;

	__m256d
		v_mask, v_left, v_zeros,
		v_alpha, v_ux, v_dux, v_pi, v_dpi, 
		v_t0, v_dt0, v_lam0, v_dlam0, v_mu0,
		v_t1, v_dt1, v_lam1, v_dlam1, v_mu1;
		
	__m256i
		i_mask;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_zeros = _mm256_setzero_pd();
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();
	u_mu0 = _mm_setzero_pd();
	u_mu1 = _mm_setzero_pd();

	double
		*ptr_pi, *ptr_dpi, *ptr_ux, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lam, *ptr_dlam;


	// first stage
	jj = 0;

	ptr_ux   = ux[jj];
	ptr_dux  = dux[jj];
	ptr_t    = t[jj];
	ptr_dt   = dt[jj];
	ptr_lam  = lam[jj];
	ptr_dlam = dlam[jj];
	
	// update inputs
	// box constraints
	ll = 0;
	for(; ll<nbu-3; ll+=4)
		{
		v_ux    = _mm256_load_pd( &ptr_ux[ll] );
		v_dux   = _mm256_load_pd( &ptr_dux[ll] );
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
		v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
		v_dux   = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
		v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
		v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
		v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
		v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
		v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_dux   = _mm256_mul_pd( v_alpha, v_dux );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_ux    = _mm256_add_pd( v_ux, v_dux );
#endif
		_mm256_store_pd( &ptr_t[ll], v_t0 );
		_mm256_store_pd( &ptr_t[pnb+ll], v_t1 );
		_mm256_store_pd( &ptr_lam[ll], v_lam0 );
		_mm256_store_pd( &ptr_lam[pnb+ll], v_lam1 );
		_mm256_store_pd( &ptr_ux[ll], v_ux );
#if defined(TARGET_X64_AVX2)
		v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
		v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
		}
	if(ll<nbu && nbu==nu)
		{
		ll_left = nbu-ll;
		v_left  = _mm256_broadcast_sd( &ll_left );
		v_mask  = _mm256_loadu_pd( d_mask );
		v_mask  = _mm256_sub_pd( v_mask, v_left );
		i_mask  = _mm256_castpd_si256( v_mask );

		v_ux    = _mm256_load_pd( &ptr_ux[ll] );
		v_dux   = _mm256_load_pd( &ptr_dux[ll] );
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
		v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
		v_dux   = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
		v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
		v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
		v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
		v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
		v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_dux   = _mm256_mul_pd( v_alpha, v_dux );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_ux    = _mm256_add_pd( v_ux, v_dux );
#endif
		_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
		_mm256_maskstore_pd( &ptr_t[pnb+ll], i_mask, v_t1 );
		_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
		_mm256_maskstore_pd( &ptr_lam[pnb+ll], i_mask, v_lam1 );
		_mm256_maskstore_pd( &ptr_ux[ll], i_mask, v_ux );
#if defined(TARGET_X64_AVX2)
		v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
		v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
		v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
		v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
		v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
		}
	else
		{
		// backup ll
		ll_bkp = ll;
		// clean up inputs
		for(; ll<nu-3; ll+=4)
			{
			v_ux  = _mm256_load_pd( &ptr_ux[ll] );
			v_dux = _mm256_load_pd( &ptr_dux[ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_ux  = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_store_pd( &ptr_ux[ll], v_ux );
			}
		if(ll<nu)
			{
			ll_left = nu-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_ux  = _mm256_load_pd( &ptr_ux[ll] );
			v_dux = _mm256_load_pd( &ptr_dux[ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_ux  = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_maskstore_pd( &ptr_ux[ll], i_mask, v_ux );
			}
		// cleanup box constraints
		ll = ll_bkp;
		for(; ll<nbu-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_store_pd( &ptr_t[ll], v_t0 );
			_mm256_store_pd( &ptr_t[pnb+ll], v_t1 );
			_mm256_store_pd( &ptr_lam[ll], v_lam0 );
			_mm256_store_pd( &ptr_lam[pnb+ll], v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<nbu)
			{
			ll_left = nbu-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[pnb+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[pnb+ll], i_mask, v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		}

	// middle stage
	for(jj=1; jj<N; jj++)
		{
		
		ptr_pi   = pi[jj];
		ptr_dpi  = dpi[jj];
		ptr_ux   = ux[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

		ll = 0;
		for(; ll<nx-3; ll+=4)
			{
			v_pi  = _mm256_load_pd( &ptr_pi[ll] );
			v_dpi = _mm256_load_pd( &ptr_dpi[ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
#if defined(TARGET_X64_AVX2)
			v_pi  = _mm256_fmadd_pd( v_alpha, v_dpi, v_pi );
#else
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
#endif
			_mm256_store_pd( &ptr_pi[ll], v_pi );
			}
		if(ll<nx)
			{
			ll_left = nx-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_pi  = _mm256_load_pd( &ptr_pi[ll] );
			v_dpi = _mm256_load_pd( &ptr_dpi[ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
#if defined(TARGET_X64_AVX2)
			v_pi  = _mm256_fmadd_pd( v_alpha, v_dpi, v_pi );
#else
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
#endif
			_mm256_maskstore_pd( &ptr_pi[ll], i_mask, v_pi );
			}

		// update inputs & states
		// box constraints
		ll = 0;
		for(; ll<nb-3; ll+=4)
			{
			v_ux    = _mm256_load_pd( &ptr_ux[ll] );
			v_dux   = _mm256_load_pd( &ptr_dux[ll] );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
			v_dux   = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_dux   = _mm256_mul_pd( v_alpha, v_dux );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_ux    = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_store_pd( &ptr_t[ll], v_t0 );
			_mm256_store_pd( &ptr_t[pnb+ll], v_t1 );
			_mm256_store_pd( &ptr_lam[ll], v_lam0 );
			_mm256_store_pd( &ptr_lam[pnb+ll], v_lam1 );
			_mm256_store_pd( &ptr_ux[ll], v_ux );
#if defined(TARGET_X64_AVX2)
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<nb && nb==nx+nu)
			{
			ll_left = nb-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_ux    = _mm256_load_pd( &ptr_ux[ll] );
			v_dux   = _mm256_load_pd( &ptr_dux[ll] );
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
			v_dux   = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_dux   = _mm256_mul_pd( v_alpha, v_dux );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_ux    = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[pnb+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[pnb+ll], i_mask, v_lam1 );
			_mm256_maskstore_pd( &ptr_ux[ll], i_mask, v_ux );
#if defined(TARGET_X64_AVX2)
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		else
			{
			// backup ll
			ll_bkp = ll;
			// clean up inputs & states
			for(; ll<nu+nx-3; ll+=4)
				{
				v_ux  = _mm256_load_pd( &ptr_ux[ll] );
				v_dux = _mm256_load_pd( &ptr_dux[ll] );
				v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
				v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
				v_dux = _mm256_mul_pd( v_alpha, v_dux );
				v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
				_mm256_store_pd( &ptr_ux[ll], v_ux );
				}
			if(ll<nu+nx)
				{
				ll_left = nu+nx-ll;
				v_left= _mm256_broadcast_sd( &ll_left );
				v_mask= _mm256_loadu_pd( d_mask );
				i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

				v_ux  = _mm256_load_pd( &ptr_ux[ll] );
				v_dux = _mm256_load_pd( &ptr_dux[ll] );
				v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
				v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
				v_dux = _mm256_mul_pd( v_alpha, v_dux );
				v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
				_mm256_maskstore_pd( &ptr_ux[ll], i_mask, v_ux );
				}
			// cleanup box constraints
			ll = ll_bkp;
			for(; ll<nb-3; ll+=4)
				{
				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
				v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
				v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
				v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
				v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
				v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
				v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
				v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
				v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
				v_t0    = _mm256_add_pd( v_t0, v_dt0 );
				v_t1    = _mm256_add_pd( v_t1, v_dt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
				_mm256_store_pd( &ptr_t[ll], v_t0 );
				_mm256_store_pd( &ptr_t[pnb+ll], v_t1 );
				_mm256_store_pd( &ptr_lam[ll], v_lam0 );
				_mm256_store_pd( &ptr_lam[pnb+ll], v_lam1 );
#if defined(TARGET_X64_AVX2)
				v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
				v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
				v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
				v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
				v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
				v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
				}
			if(ll<nb)
				{
				ll_left = nb-ll;
				v_left  = _mm256_broadcast_sd( &ll_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				v_mask  = _mm256_sub_pd( v_mask, v_left );
				i_mask  = _mm256_castpd_si256( v_mask );

				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
				v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
				v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
				v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
				v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
				v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
				v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
				v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
				v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
				v_t0    = _mm256_add_pd( v_t0, v_dt0 );
				v_t1    = _mm256_add_pd( v_t1, v_dt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
				_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
				_mm256_maskstore_pd( &ptr_t[pnb+ll], i_mask, v_t1 );
				_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
				_mm256_maskstore_pd( &ptr_lam[pnb+ll], i_mask, v_lam1 );
#if defined(TARGET_X64_AVX2)
				v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
				v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
				v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
				v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
				v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
				v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
				v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
				v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
				v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
				v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
				}
			}

		}

	// last stage
	jj = N;

	ptr_pi   = pi[jj];
	ptr_dpi  = dpi[jj];
	ptr_ux   = ux[jj];
	ptr_dux  = dux[jj];
	ptr_t    = t[jj];
	ptr_dt   = dt[jj];
	ptr_lam  = lam[jj];
	ptr_dlam = dlam[jj];

	// update equality constrained multipliers
	ll = 0;
	for(; ll<nx-3; ll+=4)
		{
		v_pi  = _mm256_load_pd( &ptr_pi[ll] );
		v_dpi = _mm256_load_pd( &ptr_dpi[ll] );
		v_dpi = _mm256_sub_pd( v_dpi, v_pi );
#if defined(TARGET_X64_AVX2)
		v_pi  = _mm256_fmadd_pd( v_alpha, v_dpi, v_pi );
#else
		v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
		v_pi  = _mm256_add_pd( v_pi, v_dpi );
#endif
		_mm256_store_pd( &ptr_pi[ll], v_pi );
		}
	if(ll<nx)
		{
		ll_left = nx-ll;
		v_left= _mm256_broadcast_sd( &ll_left );
		v_mask= _mm256_loadu_pd( d_mask );
		i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

		v_pi  = _mm256_load_pd( &ptr_pi[ll] );
		v_dpi = _mm256_load_pd( &ptr_dpi[ll] );
		v_dpi = _mm256_sub_pd( v_dpi, v_pi );
#if defined(TARGET_X64_AVX2)
		v_pi  = _mm256_fmadd_pd( v_alpha, v_dpi, v_pi );
#else
		v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
		v_pi  = _mm256_add_pd( v_pi, v_dpi );
#endif
		_mm256_maskstore_pd( &ptr_pi[ll], i_mask, v_pi );
		}
	// cleanup at the beginning
	ll = nu;
	ll_end = ((nu+bs-1)/bs)*bs;
	if(nb<ll_end)
		ll_end = nb;
	for(; ll<ll_end; ll++)
		{
		u_ux   = _mm_load_sd( &ptr_ux[ll] );
		u_dux  = _mm_load_sd( &ptr_dux[ll] );
		u_t0   = _mm_load_sd( &ptr_t[ll] );
		u_t1   = _mm_load_sd( &ptr_t[pnb+ll] );
		u_lam0 = _mm_load_sd( &ptr_lam[ll] );
		u_lam1 = _mm_load_sd( &ptr_lam[pnb+ll] );
		u_dt0  = _mm_load_sd( &ptr_dt[ll] );
		u_dt1  = _mm_load_sd( &ptr_dt[pnb+ll] );
		u_dlam0= _mm_load_sd( &ptr_dlam[ll] );
		u_dlam1= _mm_load_sd( &ptr_dlam[pnb+ll] );
		u_dux  = _mm_sub_sd( u_dux, u_ux );
#if defined(TARGET_X64_AVX2)
		u_t0   = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dt0, u_t0 );
		u_t1   = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dt1, u_t1 );
		u_lam0 = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dlam0, u_lam0 );
		u_lam1 = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dlam1, u_lam1 );
		u_ux   = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dux, u_ux );
#else
		u_dt0  = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dt0 );
		u_dt1  = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dt1 );
		u_dlam0= _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dlam0 );
		u_dlam1= _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dlam1 );
		u_dux  = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_t0   = _mm_add_sd( u_t0, u_dt0 );
		u_t1   = _mm_add_sd( u_t1, u_dt1 );
		u_lam0 = _mm_add_sd( u_lam0, u_dlam0 );
		u_lam1 = _mm_add_sd( u_lam1, u_dlam1 );
		u_ux   = _mm_add_sd( u_ux, u_dux );
#endif
		_mm_store_sd( &ptr_t[ll], u_t0 );
		_mm_store_sd( &ptr_t[pnb+ll], u_t1 );
		_mm_store_sd( &ptr_lam[ll], u_lam0 );
		_mm_store_sd( &ptr_lam[pnb+ll], u_lam1 );
		_mm_store_sd( &ptr_ux[ll], u_ux );
		u_lam0 = _mm_mul_sd( u_lam0, u_t0 );
		u_lam1 = _mm_mul_sd( u_lam1, u_t1 );
		u_mu0  = _mm_add_sd( u_mu0, u_lam0 );
		u_mu1  = _mm_add_sd( u_mu1, u_lam1 );
		}
	ll_end = ((nu+bs-1)/bs)*bs;
	if(nx+nu<ll_end)
		ll_end = nx+nu;
	for(; ll<ll_end; ll++)
		{
		u_ux  = _mm_load_sd( &ptr_ux[ll] );
		u_dux = _mm_load_sd( &ptr_dux[ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
#if defined(TARGET_X64_AVX2)
		u_ux  = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dux, u_ux );
#else
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
#endif
		_mm_store_sd( &ptr_ux[ll], u_ux );
		}
	// update states
	// box constraints
	for(; ll<nb-3; ll+=4)
		{
		v_ux    = _mm256_load_pd( &ptr_ux[ll] );
		v_dux   = _mm256_load_pd( &ptr_dux[ll] );
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
		v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
		v_dux   = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
		v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
		v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
		v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
		v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
		v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_dux   = _mm256_mul_pd( v_alpha, v_dux );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_ux    = _mm256_add_pd( v_ux, v_dux );
#endif
		_mm256_store_pd( &ptr_t[ll], v_t0 );
		_mm256_store_pd( &ptr_t[pnb+ll], v_t1 );
		_mm256_store_pd( &ptr_lam[ll], v_lam0 );
		_mm256_store_pd( &ptr_lam[pnb+ll], v_lam1 );
		_mm256_store_pd( &ptr_ux[ll], v_ux );
#if defined(TARGET_X64_AVX2)
		v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
		v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
		}
	if(ll<nb && nb==nx+nu)
		{

		ll_left = nb-ll;
		v_left  = _mm256_broadcast_sd( &ll_left );
		v_mask  = _mm256_loadu_pd( d_mask );
		v_mask  = _mm256_sub_pd( v_mask, v_left );
		i_mask  = _mm256_castpd_si256( v_mask );

		v_ux    = _mm256_load_pd( &ptr_ux[ll] );
		v_dux   = _mm256_load_pd( &ptr_dux[ll] );
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
		v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
		v_dux   = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
		v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
		v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
		v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
		v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
		v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_dux   = _mm256_mul_pd( v_alpha, v_dux );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_ux    = _mm256_add_pd( v_ux, v_dux );
#endif
		_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
		_mm256_maskstore_pd( &ptr_t[pnb+ll], i_mask, v_t1 );
		_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
		_mm256_maskstore_pd( &ptr_lam[pnb+ll], i_mask, v_lam1 );
		_mm256_maskstore_pd( &ptr_ux[ll], i_mask, v_ux );
#if defined(TARGET_X64_AVX2)
		v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
		v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
		v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
		v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
		v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif

		}
	else
		{

		// backup ll
		ll_bkp = ll;
		// clean up inputs & states
		for(; ll<nu+nx-3; ll+=4)
			{
			v_ux  = _mm256_load_pd( &ptr_ux[ll] );
			v_dux = _mm256_load_pd( &ptr_dux[ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_store_pd( &ptr_ux[ll], v_ux );
			}
		if(ll<nu+nx)
			{
			ll_left = nu+nx-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_ux  = _mm256_load_pd( &ptr_ux[ll] );
			v_dux = _mm256_load_pd( &ptr_dux[ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_maskstore_pd( &ptr_ux[ll], i_mask, v_ux );
			}
		// cleanup box constraints
		ll = ll_bkp;
		for(; ll<nb-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_store_pd( &ptr_t[ll], v_t0 );
			_mm256_store_pd( &ptr_t[pnb+ll], v_t1 );
			_mm256_store_pd( &ptr_lam[ll], v_lam0 );
			_mm256_store_pd( &ptr_lam[pnb+ll], v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<nb)
			{
			ll_left = nb-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[pnb+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[pnb+ll], i_mask, v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		}

	// genreal constraints
	if(ng>0)
		{

		for(jj=0; jj<N; jj++)
			{

			ptr_t    = t[jj];
			ptr_dt   = dt[jj];
			ptr_lam  = lam[jj];
			ptr_dlam = dlam[jj];

			for(ll=2*pnb; ll<2*pnb+ng-3; ll+=4)
				{
				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_load_pd( &ptr_dt[png+ll] );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
#if defined(TARGET_X64_AVX2)
				v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
				v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
				v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
				v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
				v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
				v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
				v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
				v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
				v_t0    = _mm256_add_pd( v_t0, v_dt0 );
				v_t1    = _mm256_add_pd( v_t1, v_dt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
				_mm256_store_pd( &ptr_t[ll], v_t0 );
				_mm256_store_pd( &ptr_t[png+ll], v_t1 );
				_mm256_store_pd( &ptr_lam[ll], v_lam0 );
				_mm256_store_pd( &ptr_lam[png+ll], v_lam1 );
#if defined(TARGET_X64_AVX2)
				v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
				v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
				v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
				v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
				v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
				v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
				}
			if(ll<2*pnb+ng)
				{

				ll_left = 2*pnb+ng-ll;
				v_left  = _mm256_broadcast_sd( &ll_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				v_mask  = _mm256_sub_pd( v_mask, v_left );
				i_mask  = _mm256_castpd_si256( v_mask );

				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_load_pd( &ptr_dt[png+ll] );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
#if defined(TARGET_X64_AVX2)
				v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
				v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
				v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
				v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
				v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
				v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
				v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
				v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
				v_t0    = _mm256_add_pd( v_t0, v_dt0 );
				v_t1    = _mm256_add_pd( v_t1, v_dt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
				_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
				_mm256_maskstore_pd( &ptr_t[png+ll], i_mask, v_t1 );
				_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
				_mm256_maskstore_pd( &ptr_lam[png+ll], i_mask, v_lam1 );
#if defined(TARGET_X64_AVX2)
				v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
				v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
				v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
				v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
				v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
				v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
				v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
				v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
				v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
				v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif

				}
			}

		}
	if(ngN>0)
		{

		ptr_t    = t[N];
		ptr_dt   = dt[N];
		ptr_lam  = lam[N];
		ptr_dlam = dlam[N];

		for(ll=2*pnb; ll<2*pnb+ngN-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pngN+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pngN+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pngN+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pngN+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_store_pd( &ptr_t[ll], v_t0 );
			_mm256_store_pd( &ptr_t[pngN+ll], v_t1 );
			_mm256_store_pd( &ptr_lam[ll], v_lam0 );
			_mm256_store_pd( &ptr_lam[pngN+ll], v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<2*pnb+ngN)
			{

			ll_left = 2*pnb+ngN-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pngN+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pngN+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pngN+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pngN+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_maskstore_pd( &ptr_t[ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[pngN+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[pngN+ll], i_mask, v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif

			}

		}



	v_mu0 = _mm256_add_pd( v_mu0, v_mu1 );
	u_mu0 = _mm_add_pd( u_mu0, u_mu1 );
	u_tmp = _mm256_extractf128_pd( v_mu0, 0x1 );
	u_mu0 = _mm_add_pd( u_mu0, _mm256_castpd256_pd128( v_mu0 ) );
	u_mu0 = _mm_add_pd( u_mu0, u_tmp );
	u_mu0 = _mm_hadd_pd( u_mu0, u_mu0 );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu0 = _mm_mul_sd( u_mu0, u_tmp );
	_mm_store_sd( ptr_mu, u_mu0 );
		

	return;
	
	}


void d_update_var_soft_mpc(int N, int nx, int nu, int nh, int ns, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{

	int nb = nh + ns;

	const int nhu = nu<nh ? nu : nh ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	int jj, ll, ll_bkp, ll_end;
	double ll_left;

	double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
	
	__m128d
		u_ux, u_dux, u_pi, u_dpi, u_t, u_dt, u_lam, u_dlam, u_mu, u_tmp;

	__m256d
		v_mask, v_left,
		v_t0, v_dt0, v_lam0, v_dlam0, v_t1, v_dt1, v_lam1, v_dlam1, 
		v_alpha, v_ux, v_dux, v_pi, v_dpi, v_mu0, v_mu1;
	
	__m256i
		i_mask;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();
	u_mu = _mm_setzero_pd();



	// first stage
	jj = 0;
	
	ll = 0;
	// update inputs
	for(; ll<nu-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		}
	if(ll<nu)
		{
		ll_left = nu-ll;
		v_left= _mm256_broadcast_sd( &ll_left );
		v_mask= _mm256_loadu_pd( d_mask );
		i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_maskstore_pd( &ux[jj][ll], i_mask, v_ux );
		}
	// box constraints
	ll = 0;
	for(; ll<nhu-1; ll+=2)
		{
		v_t0    = _mm256_load_pd( &t[jj][2*ll] );
		v_lam0  = _mm256_load_pd( &lam[jj][2*ll] );
		v_dt0   = _mm256_load_pd( &dt[jj][2*ll] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		_mm256_store_pd( &t[jj][2*ll], v_t0 );
		_mm256_store_pd( &lam[jj][2*ll], v_lam0 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		}
	if(ll<nhu)
		{
		u_t    = _mm_load_pd( &t[jj][2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][2*ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][2*ll], u_t );
		_mm_store_pd( &lam[jj][2*ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{
		// update equality constrained multipliers
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
		if(ll<nx)
			{
			ll_left = nx-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
			v_pi  = _mm256_load_pd( &pi[jj][ll] );
			v_dpi = _mm256_load_pd( &dpi[jj][ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
			_mm256_maskstore_pd( &pi[jj][ll], i_mask, v_pi );
			}
		// update inputs & states
		// box constraints
		ll = 0;
		for(; ll<nb-3; ll+=4)
			{

			v_ux    = _mm256_load_pd( &ux[jj][ll] );
			v_dux   = _mm256_load_pd( &dux[jj][ll] );
			v_t0    = _mm256_load_pd( &t[jj][2*ll+0] );
			v_t1    = _mm256_load_pd( &t[jj][2*ll+4] );
			v_lam0  = _mm256_load_pd( &lam[jj][2*ll+0] );
			v_lam1  = _mm256_load_pd( &lam[jj][2*ll+4] );
			v_dt0   = _mm256_load_pd( &dt[jj][2*ll+0] );
			v_dt1   = _mm256_load_pd( &dt[jj][2*ll+4] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll+0] );
			v_dlam1 = _mm256_load_pd( &dlam[jj][2*ll+4] );
			v_dux   = _mm256_sub_pd( v_dux, v_ux );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_dux   = _mm256_mul_pd( v_alpha, v_dux );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_ux    = _mm256_add_pd( v_ux, v_dux );
			_mm256_store_pd( &t[jj][2*ll+0], v_t0 );
			_mm256_store_pd( &t[jj][2*ll+4], v_t1 );
			_mm256_store_pd( &lam[jj][2*ll+0], v_lam0 );
			_mm256_store_pd( &lam[jj][2*ll+4], v_lam1 );
			_mm256_store_pd( &ux[jj][ll], v_ux );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
			}
		// backup ll
		ll_bkp = ll;
		// clean up inputs & states
		for(; ll<nu+nx-3; ll+=4)
			{
			v_ux  = _mm256_load_pd( &ux[jj][ll] );
			v_dux = _mm256_load_pd( &dux[jj][ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
			_mm256_store_pd( &ux[jj][ll], v_ux );
			}
		if(ll<nu+nx)
			{
			ll_left = nu+nx-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
			v_ux  = _mm256_load_pd( &ux[jj][ll] );
			v_dux = _mm256_load_pd( &dux[jj][ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
			_mm256_maskstore_pd( &ux[jj][ll], i_mask, v_ux );
			}
		// cleanup box constraints
		ll = ll_bkp;
		for(; ll<nb-1; ll+=2)
			{
			v_t0    = _mm256_load_pd( &t[jj][2*ll] );
			v_lam0  = _mm256_load_pd( &lam[jj][2*ll] );
			v_dt0   = _mm256_load_pd( &dt[jj][2*ll] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			_mm256_store_pd( &t[jj][2*ll], v_t0 );
			_mm256_store_pd( &lam[jj][2*ll], v_lam0 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			}
		if(ll<nb)
			{
			u_t    = _mm_load_pd( &t[jj][2*ll] );
			u_lam  = _mm_load_pd( &lam[jj][2*ll] );
			u_dt   = _mm_load_pd( &dt[jj][2*ll] );
			u_dlam = _mm_load_pd( &dlam[jj][2*ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			_mm_store_pd( &t[jj][2*ll], u_t );
			_mm_store_pd( &lam[jj][2*ll], u_lam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}
		// soft constraints on states
		ll = nh;
		if(ll%2==1)
			{
			u_t    = _mm_load_pd( &t[jj][pnb+2*ll] );
			u_lam  = _mm_load_pd( &lam[jj][pnb+2*ll] );
			u_dt   = _mm_load_pd( &dt[jj][pnb+2*ll] );
			u_dlam = _mm_load_pd( &dlam[jj][pnb+2*ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			_mm_store_pd( &t[jj][pnb+2*ll], u_t );
			_mm_store_pd( &lam[jj][pnb+2*ll], u_lam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );

			ll++;
			}
		for(; ll<nb-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &t[jj][pnb+2*ll+0] );
			v_t1    = _mm256_load_pd( &t[jj][pnb+2*ll+4] );
			v_lam0  = _mm256_load_pd( &lam[jj][pnb+2*ll+0] );
			v_lam1  = _mm256_load_pd( &lam[jj][pnb+2*ll+4] );
			v_dt0   = _mm256_load_pd( &dt[jj][pnb+2*ll+0] );
			v_dt1   = _mm256_load_pd( &dt[jj][pnb+2*ll+4] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][pnb+2*ll+0] );
			v_dlam1 = _mm256_load_pd( &dlam[jj][pnb+2*ll+4] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			_mm256_store_pd( &t[jj][pnb+2*ll+0], v_t0 );
			_mm256_store_pd( &t[jj][pnb+2*ll+4], v_t1 );
			_mm256_store_pd( &lam[jj][pnb+2*ll+0], v_lam0 );
			_mm256_store_pd( &lam[jj][pnb+2*ll+4], v_lam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
			}
		for(; ll<nb-1; ll+=2)
			{
			v_t0    = _mm256_load_pd( &t[jj][pnb+2*ll] );
			v_lam0  = _mm256_load_pd( &lam[jj][pnb+2*ll] );
			v_dt0   = _mm256_load_pd( &dt[jj][pnb+2*ll] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][pnb+2*ll] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			_mm256_store_pd( &t[jj][pnb+2*ll], v_t0 );
			_mm256_store_pd( &lam[jj][pnb+2*ll], v_lam0 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			}
		for(; ll<nb; ll++)
			{
			u_t    = _mm_load_pd( &t[jj][pnb+2*ll] );
			u_lam  = _mm_load_pd( &lam[jj][pnb+2*ll] );
			u_dt   = _mm_load_pd( &dt[jj][pnb+2*ll] );
			u_dlam = _mm_load_pd( &dlam[jj][pnb+2*ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			_mm_store_pd( &t[jj][pnb+2*ll], u_t );
			_mm_store_pd( &lam[jj][pnb+2*ll], u_lam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}
		}

	// last stage
	jj = N;
	// update equality constrained multipliers
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
	if(ll<nx)
		{
		ll_left = nx-ll;
		v_left= _mm256_broadcast_sd( &ll_left );
		v_mask= _mm256_loadu_pd( d_mask );
		i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
		v_pi  = _mm256_load_pd( &pi[jj][ll] );
		v_dpi = _mm256_load_pd( &dpi[jj][ll] );
		v_dpi = _mm256_sub_pd( v_dpi, v_pi );
		v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
		v_pi  = _mm256_add_pd( v_pi, v_dpi );
		_mm256_maskstore_pd( &pi[jj][ll], i_mask, v_pi );
		}
	// cleanup at the beginning
	ll = nu;
	ll_end = ((nu+bs-1)/bs)*bs;
	if(nb<ll_end)
		ll_end = nb;
	for(; ll<ll_end; ll++)
		{
		u_ux   = _mm_load_sd( &ux[jj][ll] );
		u_dux  = _mm_load_sd( &dux[jj][ll] );
		u_t    = _mm_load_pd( &t[jj][2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][2*ll] );
		u_dux  = _mm_sub_sd( u_dux, u_ux );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_dux  = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		u_ux   = _mm_add_sd( u_ux, u_dux );
		_mm_store_pd( &t[jj][2*ll], u_t );
		_mm_store_pd( &lam[jj][2*ll], u_lam );
		_mm_store_sd( &ux[jj][ll], u_ux );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
	ll_end = ((nu+bs-1)/bs)*bs;
	if(nx+nu<ll_end)
		ll_end = nx+nu;
	for(; ll<ll_end; ll++)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		}
	// update inputs & states
	// box constraints
	for(; ll<nb-3; ll+=4)
		{

		v_ux    = _mm256_load_pd( &ux[jj][ll] );
		v_dux   = _mm256_load_pd( &dux[jj][ll] );
		v_t0    = _mm256_load_pd( &t[jj][2*ll+0] );
		v_t1    = _mm256_load_pd( &t[jj][2*ll+4] );
		v_lam0  = _mm256_load_pd( &lam[jj][2*ll+0] );
		v_lam1  = _mm256_load_pd( &lam[jj][2*ll+4] );
		v_dt0   = _mm256_load_pd( &dt[jj][2*ll+0] );
		v_dt1   = _mm256_load_pd( &dt[jj][2*ll+4] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll+0] );
		v_dlam1 = _mm256_load_pd( &dlam[jj][2*ll+4] );
		v_dux   = _mm256_sub_pd( v_dux, v_ux );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_dux   = _mm256_mul_pd( v_alpha, v_dux );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_ux    = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &t[jj][2*ll+0], v_t0 );
		_mm256_store_pd( &t[jj][2*ll+4], v_t1 );
		_mm256_store_pd( &lam[jj][2*ll+0], v_lam0 );
		_mm256_store_pd( &lam[jj][2*ll+4], v_lam1 );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
		}
	// backup ll
	ll_bkp = ll;
	// clean up inputs & states
	for(; ll<nu+nx-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		}
	if(ll<nu+nx)
		{
		ll_left = nu+nx-ll;
		v_left= _mm256_broadcast_sd( &ll_left );
		v_mask= _mm256_loadu_pd( d_mask );
		i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_maskstore_pd( &ux[jj][ll], i_mask, v_ux );
		}
	// cleanup box constraints
	ll = ll_bkp;
	for(; ll<nb-1; ll+=2)
		{
		v_t0    = _mm256_load_pd( &t[jj][2*ll] );
		v_lam0  = _mm256_load_pd( &lam[jj][2*ll] );
		v_dt0   = _mm256_load_pd( &dt[jj][2*ll] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		_mm256_store_pd( &t[jj][2*ll], v_t0 );
		_mm256_store_pd( &lam[jj][2*ll], v_lam0 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		}
	if(ll<nb)
		{
		u_t    = _mm_load_pd( &t[jj][2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][2*ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][2*ll], u_t );
		_mm_store_pd( &lam[jj][2*ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
	// soft constraints on states
	ll = nh;
	if(ll%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][pnb+2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][pnb+2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][pnb+2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][pnb+2*ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][pnb+2*ll], u_t );
		_mm_store_pd( &lam[jj][pnb+2*ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );

		ll++;
		}
	for(; ll<nb-1; ll+=2)
		{
		v_t0    = _mm256_load_pd( &t[jj][pnb+2*ll] );
		v_lam0  = _mm256_load_pd( &lam[jj][pnb+2*ll] );
		v_dt0   = _mm256_load_pd( &dt[jj][pnb+2*ll] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][pnb+2*ll] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		_mm256_store_pd( &t[jj][pnb+2*ll], v_t0 );
		_mm256_store_pd( &lam[jj][pnb+2*ll], v_lam0 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		}
	for(; ll<nb; ll++)
		{
		u_t    = _mm_load_pd( &t[jj][pnb+2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][pnb+2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][pnb+2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][pnb+2*ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][pnb+2*ll], u_t );
		_mm_store_pd( &lam[jj][pnb+2*ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}

	v_mu0 = _mm256_add_pd( v_mu0, v_mu1 );
	u_tmp = _mm256_extractf128_pd( v_mu0, 0x1 );
	u_mu  = _mm_add_pd( u_mu, _mm256_castpd256_pd128( v_mu0 ) );
	u_mu  = _mm_add_pd( u_mu, u_tmp );
	u_mu  = _mm_hadd_pd( u_mu, u_mu );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu  = _mm_mul_sd( u_mu, u_tmp );
	_mm_store_sd( ptr_mu, u_mu );

	return;
	
	}



void d_update_var_diag_mpc(int N, int *nx, int *nu, int *nb, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	int jj, ll;

	int pnb;
	
	double
		*ptr_pi, *ptr_dpi, *ptr_ux, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lam, *ptr_dlam;

	double mu = 0;

	for(jj=0; jj<=N; jj++)
		{

		pnb  = bs*((nb[jj]+bs-1)/bs); // cache aligned number of box constraints

		ptr_pi   = pi[jj];
		ptr_dpi  = dpi[jj];
		ptr_ux   = ux[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

		// update inputs and states
		for(ll=0; ll<nu[jj]+nx[jj]-3; ll+=4)
			{
			ptr_ux[ll+0] += alpha*(ptr_dux[ll+0] - ptr_ux[ll+0]);
			ptr_ux[ll+1] += alpha*(ptr_dux[ll+1] - ptr_ux[ll+1]);
			ptr_ux[ll+2] += alpha*(ptr_dux[ll+2] - ptr_ux[ll+2]);
			ptr_ux[ll+3] += alpha*(ptr_dux[ll+3] - ptr_ux[ll+3]);
			}
		for(; ll<nu[jj]+nx[jj]; ll++)
			{
			ptr_ux[ll] += alpha*(ptr_dux[ll] - ptr_ux[ll]);
			}
		// update equality constrained multipliers
		for(ll=0; ll<nx[jj]-3; ll+=4)
			{
			ptr_pi[ll+0] += alpha*(ptr_dpi[ll+0] - ptr_pi[ll+0]);
			ptr_pi[ll+1] += alpha*(ptr_dpi[ll+1] - ptr_pi[ll+1]);
			ptr_pi[ll+2] += alpha*(ptr_dpi[ll+2] - ptr_pi[ll+2]);
			ptr_pi[ll+3] += alpha*(ptr_dpi[ll+3] - ptr_pi[ll+3]);
			}
		for(; ll<nx[jj]; ll++)
			{
			ptr_pi[ll] += alpha*(ptr_dpi[ll] - ptr_pi[ll]);
			}
		// box constraints
		for(ll=0; ll<nb[jj]; ll++)
			{
			ptr_lam[ll+0]   += alpha*ptr_dlam[ll+0];
			ptr_lam[ll+pnb] += alpha*ptr_dlam[ll+pnb];
			ptr_t[ll+0]   += alpha*ptr_dt[ll+0];
			ptr_t[ll+pnb] += alpha*ptr_dt[ll+pnb];
			mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+pnb] * ptr_t[ll+pnb];
			}
		}

	// scale mu
	mu *= mu_scal;

	ptr_mu[0] = mu;

	return;
	
	}



void d_compute_mu_hard_mpc_tv(int N, int *nx, int *nu, int *nb, int *ng, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	int jj, ll, ll_bkp, ll_end;
	double ll_left;
	
	double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
	
	__m128d
		u_mu0, u_tmp;

	__m256d
		v_alpha, v_mask, v_left, v_zeros,
		v_t0, v_dt0, v_lam0, v_dlam0, v_mu0, 
		v_t1, v_dt1, v_lam1, v_dlam1, v_mu1;
		
	double
		*ptr_t, *ptr_lam, *ptr_dt, *ptr_dlam;

	int nb0, pnb, ng0, png;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_zeros = _mm256_setzero_pd();
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();

	for(jj=0; jj<=N; jj++)
		{

		nb0 = nb[jj];
		pnb = (nb0+bs-1)/bs*bs;
		ng0 = ng[jj];
		png = (ng0+bs-1)/bs*bs;
		
		ptr_t    = t[jj];
		ptr_lam  = lam[jj];
		ptr_dt   = dt[jj];
		ptr_dlam = dlam[jj];

		// box constraints
		ll = 0;
		for(; ll<nb0-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<nb0)
			{
			ll_left = nb0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );

			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		// general constraints
		for(ll=2*pnb; ll<2*pnb+ng0-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[png+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<2*pnb+ng0)
			{
			ll_left = 2*pnb+ng0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );

			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[png+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}

		}

	v_mu0 = _mm256_add_pd( v_mu0, v_mu1 );
	u_mu0 = _mm_add_pd( _mm256_castpd256_pd128( v_mu0 ), _mm256_extractf128_pd( v_mu0, 0x1 ) );
	u_mu0 = _mm_hadd_pd( u_mu0, u_mu0 );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu0 = _mm_mul_sd( u_mu0, u_tmp );
	_mm_store_sd( ptr_mu, u_mu0 );

	return;

	}



void d_compute_mu_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{
	
	const int nbu = nu<nb ? nu : nb ;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	//const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!
	const int pnb  = bs*((nb+bs-1)/bs); // simd aligned number of box and soft constraints
	const int png  = bs*((ng+bs-1)/bs); // simd aligned number of box and soft constraints
	const int pngN = bs*((ngN+bs-1)/bs); // simd aligned number of box and soft constraints at last stage

	int jj, ll, ll_bkp, ll_end;
	double ll_left;
	
	double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
	
	
	__m128d
		u_tmp,
		u_t0, u_dt0, u_lam0, u_dlam0, u_mu0, 
		u_t1, u_dt1, u_lam1, u_dlam1, u_mu1; 

	__m256d
		v_alpha, v_mask, v_left, v_zeros,
		v_t0, v_dt0, v_lam0, v_dlam0, v_mu0, 
		v_t1, v_dt1, v_lam1, v_dlam1, v_mu1;
		
	double
		*ptr_t, *ptr_lam, *ptr_dt, *ptr_dlam;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_zeros = _mm256_setzero_pd();
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();
	u_mu0 = _mm_setzero_pd();
	u_mu1 = _mm_setzero_pd();


	// first stage
	jj = 0;

	ptr_t    = t[jj];
	ptr_lam  = lam[jj];
	ptr_dt   = dt[jj];
	ptr_dlam = dlam[jj];
	
	// box constraints
	ll = 0;
	for(; ll<nbu-3; ll+=4)
		{
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
		v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
		v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
		v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
		v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
		v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
		v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
		v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
		}
	if(ll<nbu)
		{
		ll_left = nbu-ll;
		v_left  = _mm256_broadcast_sd( &ll_left );
		v_mask  = _mm256_loadu_pd( d_mask );
		v_mask  = _mm256_sub_pd( v_mask, v_left );

		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
		v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
		v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
		v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
		v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
		v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
		v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
		v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
		v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
		v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
		v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
		}

	// middle stage
	for(jj=1; jj<N; jj++)
		{
		
		ptr_t    = t[jj];
		ptr_lam  = lam[jj];
		ptr_dt   = dt[jj];
		ptr_dlam = dlam[jj];

		ll = 0;
		for(; ll<nb-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<nb)
			{
			ll_left = nb-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );

			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		}

	// last stage
	jj = N;
	
	ptr_t    = t[jj];
	ptr_lam  = lam[jj];
	ptr_dt   = dt[jj];
	ptr_dlam = dlam[jj];
	
	ll = nu;
	ll_end = ((nu+bs-1)/bs)*bs;
	if(nb<ll_end)
		ll_end = nb;
	for(; ll<ll_end; ll++)
		{
		u_t0   = _mm_load_sd( &ptr_t[ll] );
		u_t1   = _mm_load_sd( &ptr_t[pnb+ll] );
		u_lam0 = _mm_load_sd( &ptr_lam[ll] );
		u_lam1 = _mm_load_sd( &ptr_lam[pnb+ll] );
		u_dt0  = _mm_load_sd( &ptr_dt[ll] );
		u_dt1  = _mm_load_sd( &ptr_dt[pnb+ll] );
		u_dlam0= _mm_load_sd( &ptr_dlam[ll] );
		u_dlam1= _mm_load_sd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
		u_t0   = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dt0, u_t0 );
		u_t1   = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dt1, u_t1 );
		u_lam0 = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dlam0, u_lam0 );
		u_lam1 = _mm_fmadd_sd( _mm256_castpd256_pd128( v_alpha ), u_dlam1, u_lam1 );
		u_lam0 = _mm_mul_sd( u_lam0, u_t0 );
		u_lam1 = _mm_mul_sd( u_lam1, u_t1 );
		u_mu0  = _mm_add_sd( u_mu0, u_lam0 );
		u_mu1  = _mm_add_sd( u_mu1, u_lam1 );
#else
		u_dt0  = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dt0 );
		u_dt1  = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dt1 );
		u_dlam0= _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dlam0 );
		u_dlam1= _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dlam1 );
		u_t0   = _mm_add_sd( u_t0, u_dt0 );
		u_t1   = _mm_add_sd( u_t1, u_dt1 );
		u_lam0 = _mm_add_sd( u_lam0, u_dlam0 );
		u_lam1 = _mm_add_sd( u_lam1, u_dlam1 );
		u_lam0 = _mm_mul_sd( u_lam0, u_t0 );
		u_lam1 = _mm_mul_sd( u_lam1, u_t1 );
		u_mu0  = _mm_add_sd( u_mu0, u_lam0 );
		u_mu1  = _mm_add_sd( u_mu1, u_lam1 );
#endif
		}
	for(; ll<nb-3; ll+=4)
		{
		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
		v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
		v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
		v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
		v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
		v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
		v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
		v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
		}
	if(ll<nb)
		{
		ll_left = nb-ll;
		v_left  = _mm256_broadcast_sd( &ll_left );
		v_mask  = _mm256_loadu_pd( d_mask );
		v_mask  = _mm256_sub_pd( v_mask, v_left );

		v_t0    = _mm256_load_pd( &ptr_t[ll] );
		v_t1    = _mm256_load_pd( &ptr_t[pnb+ll] );
		v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
		v_lam1  = _mm256_load_pd( &ptr_lam[pnb+ll] );
		v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
		v_dt1   = _mm256_load_pd( &ptr_dt[pnb+ll] );
		v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
		v_dlam1 = _mm256_load_pd( &ptr_dlam[pnb+ll] );
#if defined(TARGET_X64_AVX2)
		v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
		v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
		v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
		v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
		v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
		v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
		v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
		v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
		v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
		}
	
	// general constraints
	if(ng>0)
		{

		for(jj=0; jj<N; jj++)
			{

			ptr_t    = t[jj];
			ptr_lam  = lam[jj];
			ptr_dt   = dt[jj];
			ptr_dlam = dlam[jj];

			for(ll=2*pnb; ll<2*pnb+ng-3; ll+=4)
				{
				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_load_pd( &ptr_dt[png+ll] );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
#if defined(TARGET_X64_AVX2)
				v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
				v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
				v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
				v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
				v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
				v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
				v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
				v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
				v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
				v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
				v_t0    = _mm256_add_pd( v_t0, v_dt0 );
				v_t1    = _mm256_add_pd( v_t1, v_dt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
				v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
				v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
				v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
				v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
				}
			if(ll<2*pnb+ng)
				{
				ll_left = 2*pnb+ng-ll;
				v_left  = _mm256_broadcast_sd( &ll_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				v_mask  = _mm256_sub_pd( v_mask, v_left );

				v_t0    = _mm256_load_pd( &ptr_t[ll] );
				v_t1    = _mm256_load_pd( &ptr_t[png+ll] );
				v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
				v_lam1  = _mm256_load_pd( &ptr_lam[png+ll] );
				v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
				v_dt1   = _mm256_load_pd( &ptr_dt[png+ll] );
				v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
				v_dlam1 = _mm256_load_pd( &ptr_dlam[png+ll] );
#if defined(TARGET_X64_AVX2)
				v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
				v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
				v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
				v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
				v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
				v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
				v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
				v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
				v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
				v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
				v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
				v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
				v_t0    = _mm256_add_pd( v_t0, v_dt0 );
				v_t1    = _mm256_add_pd( v_t1, v_dt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
				v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
				v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
				v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
				v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
				v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
				v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
				}
			}

		}
	if(ngN>0)
		{

		ptr_t    = t[N];
		ptr_lam  = lam[N];
		ptr_dt   = dt[N];
		ptr_dlam = dlam[N];

		for(ll=2*pnb; ll<2*pnb+ngN-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pngN+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pngN+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pngN+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pngN+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<2*pnb+ngN)
			{
			ll_left = 2*pnb+ngN-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );

			v_t0    = _mm256_load_pd( &ptr_t[ll] );
			v_t1    = _mm256_load_pd( &ptr_t[pngN+ll] );
			v_lam0  = _mm256_load_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_load_pd( &ptr_lam[pngN+ll] );
			v_dt0   = _mm256_load_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_load_pd( &ptr_dt[pngN+ll] );
			v_dlam0 = _mm256_load_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_load_pd( &ptr_dlam[pngN+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}

		}
	
	v_mu0 = _mm256_add_pd( v_mu0, v_mu1 );
	u_mu0 = _mm_add_pd( u_mu0, u_mu1 );
	u_tmp = _mm256_extractf128_pd( v_mu0, 0x1 );
	u_mu0 = _mm_add_pd( u_mu0, _mm256_castpd256_pd128( v_mu0 ) );
	u_mu0 = _mm_add_pd( u_mu0, u_tmp );
	u_mu0 = _mm_hadd_pd( u_mu0, u_mu0 );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu0 = _mm_mul_sd( u_mu0, u_tmp );
	_mm_store_sd( ptr_mu, u_mu0 );
		

	return;

	}



void d_compute_mu_soft_mpc(int N, int nx, int nu, int nh, int ns, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{

	int nb = nh + ns;
	
	int nhu = nu<nh ? nu : nh ;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	int jj, ll;
	
	__m128d
		u_t, u_dt, u_lam, u_dlam, u_mu, u_tmp;

	__m256d
		v_alpha, v_t0, v_t1, v_dt0, v_dt1, v_lam0, v_lam1, v_dlam0, v_dlam1, v_mu0, v_mu1;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();
	u_mu = _mm_setzero_pd();

	

	// first stage
	jj = 0;
	
	// box constraints
	ll = 0;
	for(; ll<2*nhu-7; ll+=8)
		{
		v_t0    = _mm256_load_pd( &t[jj][ll+0] );
		v_t1    = _mm256_load_pd( &t[jj][ll+4] );
		v_lam0  = _mm256_load_pd( &lam[jj][ll+0] );
		v_lam1  = _mm256_load_pd( &lam[jj][ll+4] );
		v_dt0   = _mm256_load_pd( &dt[jj][ll+0] );
		v_dt1   = _mm256_load_pd( &dt[jj][ll+4] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][ll+0] );
		v_dlam1 = _mm256_load_pd( &dlam[jj][ll+4] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
		}
	for(; ll<2*nhu-3; ll+=4)
		{
		v_t0    = _mm256_load_pd( &t[jj][ll] );
		v_lam0  = _mm256_load_pd( &lam[jj][ll] );
		v_dt0   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][ll] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		}
	if(ll<2*nhu-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{
		ll = 0;
		for(; ll<2*nb-7; ll+=8)
			{
			v_t0    = _mm256_load_pd( &t[jj][ll+0] );
			v_t1    = _mm256_load_pd( &t[jj][ll+4] );
			v_lam0  = _mm256_load_pd( &lam[jj][ll+0] );
			v_lam1  = _mm256_load_pd( &lam[jj][ll+4] );
			v_dt0   = _mm256_load_pd( &dt[jj][ll+0] );
			v_dt1   = _mm256_load_pd( &dt[jj][ll+4] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][ll+0] );
			v_dlam1 = _mm256_load_pd( &dlam[jj][ll+4] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
			}
		for(; ll<2*nb-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &t[jj][ll+0] );
			v_lam0  = _mm256_load_pd( &lam[jj][ll+0] );
			v_dt0   = _mm256_load_pd( &dt[jj][ll+0] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][ll+0] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
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
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		// soft constraints
		ll = 2*nh;
		if(nh%2==1)
			{
			u_t    = _mm_load_pd( &t[jj][pnb+ll] );
			u_lam  = _mm_load_pd( &lam[jj][pnb+ll] );
			u_dt   = _mm_load_pd( &dt[jj][pnb+ll] );
			u_dlam = _mm_load_pd( &dlam[jj][pnb+ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			ll += 2;
			}
		for(; ll<2*nb-7; ll+=8)
			{
			v_t0    = _mm256_load_pd( &t[jj][pnb+ll+0] );
			v_t1    = _mm256_load_pd( &t[jj][pnb+ll+4] );
			v_lam0  = _mm256_load_pd( &lam[jj][pnb+ll+0] );
			v_lam1  = _mm256_load_pd( &lam[jj][pnb+ll+4] );
			v_dt0   = _mm256_load_pd( &dt[jj][pnb+ll+0] );
			v_dt1   = _mm256_load_pd( &dt[jj][pnb+ll+4] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][pnb+ll+0] );
			v_dlam1 = _mm256_load_pd( &dlam[jj][pnb+ll+4] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
			}
		for(; ll<2*nb-3; ll+=4)
			{
			v_t0    = _mm256_load_pd( &t[jj][pnb+ll+0] );
			v_lam0  = _mm256_load_pd( &lam[jj][pnb+ll+0] );
			v_dt0   = _mm256_load_pd( &dt[jj][pnb+ll+0] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][pnb+ll+0] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			}
		if(ll<2*nb-1)
			{
			u_t    = _mm_load_pd( &t[jj][pnb+ll] );
			u_lam  = _mm_load_pd( &lam[jj][pnb+ll] );
			u_dt   = _mm_load_pd( &dt[jj][pnb+ll] );
			u_dlam = _mm_load_pd( &dlam[jj][pnb+ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		}	

	// last stage
	jj = N;
	
	// hard constraints
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
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		ll += 2;
		}
	for(; ll<2*nb-7; ll+=8)
		{
		v_t0    = _mm256_load_pd( &t[jj][ll+0] );
		v_t1    = _mm256_load_pd( &t[jj][ll+4] );
		v_lam0  = _mm256_load_pd( &lam[jj][ll+0] );
		v_lam1  = _mm256_load_pd( &lam[jj][ll+4] );
		v_dt0   = _mm256_load_pd( &dt[jj][ll+0] );
		v_dt1   = _mm256_load_pd( &dt[jj][ll+4] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][ll+0] );
		v_dlam1 = _mm256_load_pd( &dlam[jj][ll+4] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
		}
	for(; ll<2*nb-3; ll+=4)
		{
		v_t0    = _mm256_load_pd( &t[jj][ll+0] );
		v_lam0  = _mm256_load_pd( &lam[jj][ll+0] );
		v_dt0   = _mm256_load_pd( &dt[jj][ll+0] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][ll+0] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
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
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}

	// soft constraints
	ll = 2*nh;
	if(nh%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][pnb+ll] );
		u_lam  = _mm_load_pd( &lam[jj][pnb+ll] );
		u_dt   = _mm_load_pd( &dt[jj][pnb+ll] );
		u_dlam = _mm_load_pd( &dlam[jj][pnb+ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		ll += 2;
		}
	for(; ll<2*nb-7; ll+=8)
		{
		v_t0    = _mm256_load_pd( &t[jj][pnb+ll+0] );
		v_t1    = _mm256_load_pd( &t[jj][pnb+ll+4] );
		v_lam0  = _mm256_load_pd( &lam[jj][pnb+ll+0] );
		v_lam1  = _mm256_load_pd( &lam[jj][pnb+ll+4] );
		v_dt0   = _mm256_load_pd( &dt[jj][pnb+ll+0] );
		v_dt1   = _mm256_load_pd( &dt[jj][pnb+ll+4] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][pnb+ll+0] );
		v_dlam1 = _mm256_load_pd( &dlam[jj][pnb+ll+4] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
		}
	for(; ll<2*nb-3; ll+=4)
		{
		v_t0    = _mm256_load_pd( &t[jj][pnb+ll+0] );
		v_lam0  = _mm256_load_pd( &lam[jj][pnb+ll+0] );
		v_dt0   = _mm256_load_pd( &dt[jj][pnb+ll+0] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][pnb+ll+0] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
		}
	if(ll<2*nb-1)
		{
		u_t    = _mm_load_pd( &t[jj][pnb+ll] );
		u_lam  = _mm_load_pd( &lam[jj][pnb+ll] );
		u_dt   = _mm_load_pd( &dt[jj][pnb+ll] );
		u_dlam = _mm_load_pd( &dlam[jj][pnb+ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}

	v_mu0 = _mm256_add_pd( v_mu0, v_mu1 );
	u_tmp = _mm256_extractf128_pd( v_mu0, 0x1 );
	u_mu  = _mm_add_pd( u_mu, _mm256_castpd256_pd128( v_mu0 ) );
	u_mu  = _mm_add_pd( u_mu, u_tmp );
	u_mu  = _mm_hadd_pd( u_mu, u_mu );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu  = _mm_mul_sd( u_mu, u_tmp );
	_mm_store_sd( ptr_mu, u_mu );

	return;

	}



void d_compute_mu_diag_mpc(int N, int *nx, int *nu, int *nb, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	int jj, ll;

	int pnb;
	
	double
		*ptr_t, *ptr_lam, *ptr_dt, *ptr_dlam;
		
	double mu = 0;
	
	for(jj=0; jj<=N; jj++)
		{
		
		pnb  = bs*((nb[jj]+bs-1)/bs); // simd aligned number of box and soft constraints

		ptr_t    = t[jj];
		ptr_lam  = lam[jj];
		ptr_dt   = dt[jj];
		ptr_dlam = dlam[jj];

		for(ll=0 ; ll<nb[jj]; ll++)
			{
			mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+pnb] + alpha*ptr_dlam[ll+pnb]) * (ptr_t[ll+pnb] + alpha*ptr_dt[ll+pnb]);
			}
		}

	// scale mu
	mu *= mu_scal;
		
	ptr_mu[0] = mu;

	return;

	}




