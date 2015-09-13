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

#include <math.h>
#include "../../include/blas_d.h"
#include "../../include/block_size.h"



void d_init_var_hard_mpc_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **ux, double **pi, double **pDCt, double **db, double **t, double **lam, double mu0, int warm_start)
	{

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

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



void d_init_var_soft_mpc_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, int *ns, double **ux, double **pi, double **pDCt, double **db, double **t, double **lam, double mu0, int warm_start)
	{

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int jj, ll, ii;

	int nb0, pnb, ng0, png, cng, ns0, pns;
	
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


	// inizialize t_theta and lam_theta (cold start only for the moment)
	for(jj=0; jj<=N; jj++)
		{
		nb0 = nb[jj];
		pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints
		ng0 = ng[jj];
		png  = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints
		ns0 = ns[jj];
		pns  = (ns0+bs-1)/bs*bs; // simd aligned number of box soft constraints
		for(ll=0; ll<ns[jj]; ll++)
			{
			t[jj][2*pnb+2*png+0*pns+ll] = 1.0;
			t[jj][2*pnb+2*png+1*pns+ll] = 1.0;
			t[jj][2*pnb+2*png+2*pns+ll] = 1.0;
			t[jj][2*pnb+2*png+3*pns+ll] = 1.0;
			lam[jj][2*pnb+2*png+0*pns+ll] = mu0; // /t[jj][pnb+ll]; // TODO restore division if needed
			lam[jj][2*pnb+2*png+1*pns+ll] = mu0; // /t[jj][pnb+ll]; // TODO restore division if needed
			lam[jj][2*pnb+2*png+2*pns+ll] = mu0; // /t[jj][pnb+ll]; // TODO restore division if needed
			lam[jj][2*pnb+2*png+3*pns+ll] = mu0; // /t[jj][pnb+ll]; // TODO restore division if needed
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
	const int bs  = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	//const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box constraints
	const int pnb  = bs*((nb+bs-1)/bs); // simd aligned number of box constraints
	//const int ang = nal*((ng+nal-1)/nal); // cache aligned number of general constraints
	const int png  = bs*((ng+bs-1)/bs); // cache aligned number of general constraints
	const int pngN = bs*((ngN+bs-1)/bs); // cache aligned number of general constraints at stage N
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);

	int jj, ll, ii;
	
	double
		*ptr_t, *ptr_lam, *ptr_db;

	double thr0 = 0.1; // minimum vale of t (minimum distance from a constraint)

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
/*			t[0][ll] = 1.0;*/
/*			t[0][pnb+ll] = 1.0;*/
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
/*				t[jj][ll] = 1.0;*/
/*				t[jj][pnb+ll] = 1.0;*/
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
/*			t[N][ll] = 1.0;*/
/*			t[N][pnb+ll] = 1.0;*/
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

	}



void d_init_var_soft_mpc(int N, int nx, int nu, int nh, int ns, double **ux, double **pi, double **db, double **t, double **lam, double mu0, int warm_start)
	{

	int nb = nh + ns;
	
	const int nhu = nu<nh ? nu : nh ;

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
		for(ll=0; ll<2*nhu; ll+=2)
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
		for(ll=0; ll<2*nhu; ll++) // this has to be strictly positive !!!
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
	else // cold start
		{
		// first stage
		for(ll=0; ll<2*nhu; ll+=2)
			{
			ux[0][ll/2] = 0.0;
/*			t[0][ll+0] = 1.0;*/
/*			t[0][ll+1] = 1.0;*/
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
/*				t[jj][ll+0] = 1.0;*/
/*				t[jj][ll+1] = 1.0;*/
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
		for(ll=0; ll<2*nhu; ll++)
			{
			t[N][ll] = 1.0; // this has to be strictly positive !!!
			lam[N][ll] = 1.0;
			}
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			ux[N][ll/2] = 0.0;
/*			t[N][ll+0] = 1.0;*/
/*			t[N][ll+1] = 1.0;*/
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



void d_update_hessian_hard_mpc_tv(int N, int *nx, int *nu, int *nb, int *ng, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **Qx, double **qx, double **qx2, double **bd, double **bl, double **pd, double **pl, double **db)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nb0, pnb, ng0, png;
	
	double temp0, temp1;
	
	double 
		*ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Qx, *ptr_qx, *ptr_qx2,
		*ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv;
	
	int ii, jj, bs0;
	
	for(jj=0; jj<=N; jj++)
		{
		
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

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints

			for(ii=0; ii<nb0-3; ii+=4)
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
			for(; ii<nb0; ii++)
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

			ptr_t     += 2*pnb;
			ptr_lam   += 2*pnb;
			ptr_lamt  += 2*pnb;
			ptr_dlam  += 2*pnb;
			ptr_tinv  += 2*pnb;
			ptr_db    += 2*pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			ptr_Qx    = Qx[jj];
			ptr_qx    = qx[jj];
			ptr_qx2   = qx2[jj];
		
			png = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints

			for(ii=0; ii<ng0-3; ii+=4)
				{

				ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_tinv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
				ptr_lamt[ii+png+0] = ptr_lam[ii+png+0]*ptr_tinv[ii+png+0];
				ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+0] = ptr_tinv[ii+png+0]*sigma_mu; // !!!!!
				ptr_Qx[ii+0] = sqrt(ptr_lamt[ii+0] + ptr_lamt[ii+png+0]);
				ptr_qx[ii+0] =  (ptr_lam[ii+png+0] + ptr_lamt[ii+png+0]*ptr_db[ii+png+0] + ptr_dlam[ii+png+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0]);
				ptr_qx2[ii+0] = ptr_qx[ii+0] / ptr_Qx[ii+0];

				ptr_tinv[ii+1] = 1.0/ptr_t[ii+1];
				ptr_tinv[ii+png+1] = 1.0/ptr_t[ii+png+1];
				ptr_lamt[ii+1] = ptr_lam[ii+1]*ptr_tinv[ii+1];
				ptr_lamt[ii+png+1] = ptr_lam[ii+png+1]*ptr_tinv[ii+png+1];
				ptr_dlam[ii+1] = ptr_tinv[ii+1]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+1] = ptr_tinv[ii+png+1]*sigma_mu; // !!!!!
				ptr_Qx[ii+1] = sqrt(ptr_lamt[ii+1] + ptr_lamt[ii+png+1]);
				ptr_qx[ii+1] =  (ptr_lam[ii+png+1] + ptr_lamt[ii+png+1]*ptr_db[ii+png+1] + ptr_dlam[ii+png+1] - ptr_lam[ii+1] - ptr_lamt[ii+1]*ptr_db[ii+1] - ptr_dlam[ii+1]);
				ptr_qx2[ii+1] = ptr_qx[ii+1] / ptr_Qx[ii+1];

				ptr_tinv[ii+2] = 1.0/ptr_t[ii+2];
				ptr_tinv[ii+png+2] = 1.0/ptr_t[ii+png+2];
				ptr_lamt[ii+2] = ptr_lam[ii+2]*ptr_tinv[ii+2];
				ptr_lamt[ii+png+2] = ptr_lam[ii+png+2]*ptr_tinv[ii+png+2];
				ptr_dlam[ii+2] = ptr_tinv[ii+2]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+2] = ptr_tinv[ii+png+2]*sigma_mu; // !!!!!
				ptr_Qx[ii+2] = sqrt(ptr_lamt[ii+2] + ptr_lamt[ii+png+2]);
				ptr_qx[ii+2] =  (ptr_lam[ii+png+2] + ptr_lamt[ii+png+2]*ptr_db[ii+png+2] + ptr_dlam[ii+png+2] - ptr_lam[ii+2] - ptr_lamt[ii+2]*ptr_db[ii+2] - ptr_dlam[ii+2]);
				ptr_qx2[ii+2] = ptr_qx[ii+2] / ptr_Qx[ii+2];

				ptr_tinv[ii+3] = 1.0/ptr_t[ii+3];
				ptr_tinv[ii+png+3] = 1.0/ptr_t[ii+png+3];
				ptr_lamt[ii+3] = ptr_lam[ii+3]*ptr_tinv[ii+3];
				ptr_lamt[ii+png+3] = ptr_lam[ii+png+3]*ptr_tinv[ii+png+3];
				ptr_dlam[ii+3] = ptr_tinv[ii+3]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+3] = ptr_tinv[ii+png+3]*sigma_mu; // !!!!!
				ptr_Qx[ii+3] = sqrt(ptr_lamt[ii+3] + ptr_lamt[ii+png+3]);
				ptr_qx[ii+3] =  (ptr_lam[ii+png+3] + ptr_lamt[ii+png+3]*ptr_db[ii+png+3] + ptr_dlam[ii+png+3] - ptr_lam[ii+3] - ptr_lamt[ii+3]*ptr_db[ii+3] - ptr_dlam[ii+3]);
				ptr_qx2[ii+3] = ptr_qx[ii+3] / ptr_Qx[ii+3];

				}
			for(; ii<ng0; ii++)
				{

				ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_tinv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
				ptr_lamt[ii+png+0] = ptr_lam[ii+png+0]*ptr_tinv[ii+png+0];
				ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+0] = ptr_tinv[ii+png+0]*sigma_mu; // !!!!!
				ptr_Qx[ii+0] = sqrt(ptr_lamt[ii+0] + ptr_lamt[ii+png+0]);
				ptr_qx[ii+0] =  (ptr_lam[ii+png+0] + ptr_lamt[ii+png+0]*ptr_db[ii+png+0] + ptr_dlam[ii+png+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0]);
				ptr_qx2[ii+0] = ptr_qx[ii+0] / ptr_Qx[ii+0];

				}

			}

		}

	}



void d_update_hessian_soft_mpc_tv(int N, int *nx, int *nu, int *nb, int *ng, int *ns, double sigma_mu, double **t, double **tinv, double **lam, double **lamt, double **dlam, double **Qx, double **qx, double **qx2, double **bd, double **bl, double **pd, double **pl, double **db, double **Z, double **z, double **Zl, double **zl)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nb0, pnb, ng0, png, ns0, pns;
	
	double temp0, temp1;
	
	double 
		*ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Qx, *ptr_qx, *ptr_qx2,
		*ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv,
		*ptr_Z, *ptr_z, *ptr_Zl, *ptr_zl;
	
	static double rQx[8] = {};
	static double rqx[8] = {};
	
	int ii, jj, bs0;
	
	for(jj=0; jj<=N; jj++)
		{
		
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

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints

			for(ii=0; ii<nb0-3; ii+=4)
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
			for(; ii<nb0; ii++)
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

			ptr_t     += 2*pnb;
			ptr_lam   += 2*pnb;
			ptr_lamt  += 2*pnb;
			ptr_dlam  += 2*pnb;
			ptr_tinv  += 2*pnb;
			ptr_db    += 2*pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			ptr_Qx    = Qx[jj];
			ptr_qx    = qx[jj];
			ptr_qx2   = qx2[jj];

			png = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints

			for(ii=0; ii<ng0-3; ii+=4)
				{

				ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_tinv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
				ptr_lamt[ii+png+0] = ptr_lam[ii+png+0]*ptr_tinv[ii+png+0];
				ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+0] = ptr_tinv[ii+png+0]*sigma_mu; // !!!!!
				ptr_Qx[ii+0] = sqrt(ptr_lamt[ii+0] + ptr_lamt[ii+png+0]);
				ptr_qx[ii+0] =  (ptr_lam[ii+png+0] + ptr_lamt[ii+png+0]*ptr_db[ii+png+0] + ptr_dlam[ii+png+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0]);
				ptr_qx2[ii+0] = ptr_qx[ii+0] / ptr_Qx[ii+0];

				ptr_tinv[ii+1] = 1.0/ptr_t[ii+1];
				ptr_tinv[ii+png+1] = 1.0/ptr_t[ii+png+1];
				ptr_lamt[ii+1] = ptr_lam[ii+1]*ptr_tinv[ii+1];
				ptr_lamt[ii+png+1] = ptr_lam[ii+png+1]*ptr_tinv[ii+png+1];
				ptr_dlam[ii+1] = ptr_tinv[ii+1]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+1] = ptr_tinv[ii+png+1]*sigma_mu; // !!!!!
				ptr_Qx[ii+1] = sqrt(ptr_lamt[ii+1] + ptr_lamt[ii+png+1]);
				ptr_qx[ii+1] =  (ptr_lam[ii+png+1] + ptr_lamt[ii+png+1]*ptr_db[ii+png+1] + ptr_dlam[ii+png+1] - ptr_lam[ii+1] - ptr_lamt[ii+1]*ptr_db[ii+1] - ptr_dlam[ii+1]);
				ptr_qx2[ii+1] = ptr_qx[ii+1] / ptr_Qx[ii+1];

				ptr_tinv[ii+2] = 1.0/ptr_t[ii+2];
				ptr_tinv[ii+png+2] = 1.0/ptr_t[ii+png+2];
				ptr_lamt[ii+2] = ptr_lam[ii+2]*ptr_tinv[ii+2];
				ptr_lamt[ii+png+2] = ptr_lam[ii+png+2]*ptr_tinv[ii+png+2];
				ptr_dlam[ii+2] = ptr_tinv[ii+2]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+2] = ptr_tinv[ii+png+2]*sigma_mu; // !!!!!
				ptr_Qx[ii+2] = sqrt(ptr_lamt[ii+2] + ptr_lamt[ii+png+2]);
				ptr_qx[ii+2] =  (ptr_lam[ii+png+2] + ptr_lamt[ii+png+2]*ptr_db[ii+png+2] + ptr_dlam[ii+png+2] - ptr_lam[ii+2] - ptr_lamt[ii+2]*ptr_db[ii+2] - ptr_dlam[ii+2]);
				ptr_qx2[ii+2] = ptr_qx[ii+2] / ptr_Qx[ii+2];

				ptr_tinv[ii+3] = 1.0/ptr_t[ii+3];
				ptr_tinv[ii+png+3] = 1.0/ptr_t[ii+png+3];
				ptr_lamt[ii+3] = ptr_lam[ii+3]*ptr_tinv[ii+3];
				ptr_lamt[ii+png+3] = ptr_lam[ii+png+3]*ptr_tinv[ii+png+3];
				ptr_dlam[ii+3] = ptr_tinv[ii+3]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+3] = ptr_tinv[ii+png+3]*sigma_mu; // !!!!!
				ptr_Qx[ii+3] = sqrt(ptr_lamt[ii+3] + ptr_lamt[ii+png+3]);
				ptr_qx[ii+3] =  (ptr_lam[ii+png+3] + ptr_lamt[ii+png+3]*ptr_db[ii+png+3] + ptr_dlam[ii+png+3] - ptr_lam[ii+3] - ptr_lamt[ii+3]*ptr_db[ii+3] - ptr_dlam[ii+3]);
				ptr_qx2[ii+3] = ptr_qx[ii+3] / ptr_Qx[ii+3];

				}
			for(; ii<ng0; ii++)
				{

				ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_tinv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
				ptr_lamt[ii+png+0] = ptr_lam[ii+png+0]*ptr_tinv[ii+png+0];
				ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+0] = ptr_tinv[ii+png+0]*sigma_mu; // !!!!!
				ptr_Qx[ii+0] = sqrt(ptr_lamt[ii+0] + ptr_lamt[ii+png+0]);
				ptr_qx[ii+0] =  (ptr_lam[ii+png+0] + ptr_lamt[ii+png+0]*ptr_db[ii+png+0] + ptr_dlam[ii+png+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0]);
				ptr_qx2[ii+0] = ptr_qx[ii+0] / ptr_Qx[ii+0];

				}

			ptr_t     += 2*png;
			ptr_lam   += 2*png;
			ptr_lamt  += 2*png;
			ptr_dlam  += 2*png;
			ptr_tinv  += 2*png;
			ptr_db    += 2*png;

			}

		// box soft constraints
		ns0 = ns[jj];
		if(ns0>0)
			{

			ptr_Z     = Z[jj];
			ptr_z     = z[jj];
			ptr_Zl    = Zl[jj];
			ptr_zl    = zl[jj];

			pns  = (ns0+bs-1)/bs*bs; // simd aligned number of box constraints

			for(ii=0; ii<ns0-3; ii+=4)
				{

				ptr_tinv[ii+0*pns+0] = 1.0/ptr_t[ii+0*pns+0];
				ptr_tinv[ii+1*pns+0] = 1.0/ptr_t[ii+1*pns+0];
				ptr_tinv[ii+2*pns+0] = 1.0/ptr_t[ii+2*pns+0];
				ptr_tinv[ii+3*pns+0] = 1.0/ptr_t[ii+3*pns+0];
				ptr_lamt[ii+0*pns+0] = ptr_lam[ii+0*pns+0]*ptr_tinv[ii+0*pns+0];
				ptr_lamt[ii+1*pns+0] = ptr_lam[ii+1*pns+0]*ptr_tinv[ii+1*pns+0];
				ptr_lamt[ii+2*pns+0] = ptr_lam[ii+2*pns+0]*ptr_tinv[ii+2*pns+0];
				ptr_lamt[ii+3*pns+0] = ptr_lam[ii+3*pns+0]*ptr_tinv[ii+3*pns+0];
				ptr_dlam[ii+0*pns+0] = ptr_tinv[ii+0*pns+0]*sigma_mu;
				ptr_dlam[ii+1*pns+0] = ptr_tinv[ii+1*pns+0]*sigma_mu;
				ptr_dlam[ii+2*pns+0] = ptr_tinv[ii+2*pns+0]*sigma_mu;
				ptr_dlam[ii+3*pns+0] = ptr_tinv[ii+3*pns+0]*sigma_mu;
				rQx[0] = ptr_lamt[ii+0*pns+0];
				rQx[1] = ptr_lamt[ii+1*pns+0];
				rqx[0] = ptr_lam[ii+0*pns+0] + ptr_dlam[ii+0*pns+0] + ptr_lamt[ii+0*pns+0]*ptr_db[ii+0*pns+0];
				rqx[1] = ptr_lam[ii+1*pns+0] + ptr_dlam[ii+1*pns+0] + ptr_lamt[ii+1*pns+0]*ptr_db[ii+1*pns+0];
				ptr_Zl[ii+0*pns+0] = 1.0 / (ptr_Z[ii+0*pns+0] + rQx[0] + ptr_lamt[ii+2*pns+0]);
				ptr_Zl[ii+1*pns+0] = 1.0 / (ptr_Z[ii+1*pns+0] + rQx[1] + ptr_lamt[ii+3*pns+0]);
				ptr_zl[ii+0*pns+0] = - ptr_z[ii+0*pns+0] + rqx[0] + ptr_lam[ii+2*pns+0] + ptr_dlam[ii+2*pns+0];
				ptr_zl[ii+1*pns+0] = - ptr_z[ii+1*pns+0] + rqx[1] + ptr_lam[ii+3*pns+0] + ptr_dlam[ii+3*pns+0];
				rqx[0] = rqx[0] - rQx[0]*ptr_zl[ii+0*pns+0]*ptr_Zl[ii+0*pns+0]; // update this before Qx !!!!!!!!!!!
				rqx[1] = rqx[1] - rQx[1]*ptr_zl[ii+1*pns+0]*ptr_Zl[ii+1*pns+0]; // update this before Qx !!!!!!!!!!!
				rQx[0] = rQx[0] - rQx[0]*rQx[0]*ptr_Zl[ii+0*pns+0];
				rQx[1] = rQx[1] - rQx[1]*rQx[1]*ptr_Zl[ii+1*pns+0];
				ptr_pd[nb0+ii+0] = ptr_bd[nb0+ii+0] + rQx[1] + rQx[0];
				ptr_pl[nb0+ii+0] = ptr_bl[nb0+ii+0] + rqx[1] - rqx[0];

				ptr_tinv[ii+0*pns+1] = 1.0/ptr_t[ii+0*pns+1];
				ptr_tinv[ii+1*pns+1] = 1.0/ptr_t[ii+1*pns+1];
				ptr_tinv[ii+2*pns+1] = 1.0/ptr_t[ii+2*pns+1];
				ptr_tinv[ii+3*pns+1] = 1.0/ptr_t[ii+3*pns+1];
				ptr_lamt[ii+0*pns+1] = ptr_lam[ii+0*pns+1]*ptr_tinv[ii+0*pns+1];
				ptr_lamt[ii+1*pns+1] = ptr_lam[ii+1*pns+1]*ptr_tinv[ii+1*pns+1];
				ptr_lamt[ii+2*pns+1] = ptr_lam[ii+2*pns+1]*ptr_tinv[ii+2*pns+1];
				ptr_lamt[ii+3*pns+1] = ptr_lam[ii+3*pns+1]*ptr_tinv[ii+3*pns+1];
				ptr_dlam[ii+0*pns+1] = ptr_tinv[ii+0*pns+1]*sigma_mu;
				ptr_dlam[ii+1*pns+1] = ptr_tinv[ii+1*pns+1]*sigma_mu;
				ptr_dlam[ii+2*pns+1] = ptr_tinv[ii+2*pns+1]*sigma_mu;
				ptr_dlam[ii+3*pns+1] = ptr_tinv[ii+3*pns+1]*sigma_mu;
				rQx[0] = ptr_lamt[ii+0*pns+1];
				rQx[1] = ptr_lamt[ii+1*pns+1];
				rqx[0] = ptr_lam[ii+0*pns+1] + ptr_dlam[ii+0*pns+1] + ptr_lamt[ii+0*pns+1]*ptr_db[ii+0*pns+1];
				rqx[1] = ptr_lam[ii+1*pns+1] + ptr_dlam[ii+1*pns+1] + ptr_lamt[ii+1*pns+1]*ptr_db[ii+1*pns+1];
				ptr_Zl[ii+0*pns+1] = 1.0 / (ptr_Z[ii+0*pns+1] + rQx[0] + ptr_lamt[ii+2*pns+1]);
				ptr_Zl[ii+1*pns+1] = 1.0 / (ptr_Z[ii+1*pns+1] + rQx[1] + ptr_lamt[ii+3*pns+1]);
				ptr_zl[ii+0*pns+1] = - ptr_z[ii+0*pns+1] + rqx[0] + ptr_lam[ii+2*pns+1] + ptr_dlam[ii+2*pns+1];
				ptr_zl[ii+1*pns+1] = - ptr_z[ii+1*pns+1] + rqx[1] + ptr_lam[ii+3*pns+1] + ptr_dlam[ii+3*pns+1];
				rqx[0] = rqx[0] - rQx[0]*ptr_zl[ii+0*pns+1]*ptr_Zl[ii+0*pns+1]; // update this before Qx !!!!!!!!!!!
				rqx[1] = rqx[1] - rQx[1]*ptr_zl[ii+1*pns+1]*ptr_Zl[ii+1*pns+1]; // update this before Qx !!!!!!!!!!!
				rQx[0] = rQx[0] - rQx[0]*rQx[0]*ptr_Zl[ii+0*pns+1];
				rQx[1] = rQx[1] - rQx[1]*rQx[1]*ptr_Zl[ii+1*pns+1];
				ptr_pd[nb0+ii+1] = ptr_bd[nb0+ii+1] + rQx[1] + rQx[0];
				ptr_pl[nb0+ii+1] = ptr_bl[nb0+ii+1] + rqx[1] - rqx[0];

				ptr_tinv[ii+0*pns+2] = 1.0/ptr_t[ii+0*pns+2];
				ptr_tinv[ii+1*pns+2] = 1.0/ptr_t[ii+1*pns+2];
				ptr_tinv[ii+2*pns+2] = 1.0/ptr_t[ii+2*pns+2];
				ptr_tinv[ii+3*pns+2] = 1.0/ptr_t[ii+3*pns+2];
				ptr_lamt[ii+0*pns+2] = ptr_lam[ii+0*pns+2]*ptr_tinv[ii+0*pns+2];
				ptr_lamt[ii+1*pns+2] = ptr_lam[ii+1*pns+2]*ptr_tinv[ii+1*pns+2];
				ptr_lamt[ii+2*pns+2] = ptr_lam[ii+2*pns+2]*ptr_tinv[ii+2*pns+2];
				ptr_lamt[ii+3*pns+2] = ptr_lam[ii+3*pns+2]*ptr_tinv[ii+3*pns+2];
				ptr_dlam[ii+0*pns+2] = ptr_tinv[ii+0*pns+2]*sigma_mu;
				ptr_dlam[ii+1*pns+2] = ptr_tinv[ii+1*pns+2]*sigma_mu;
				ptr_dlam[ii+2*pns+2] = ptr_tinv[ii+2*pns+2]*sigma_mu;
				ptr_dlam[ii+3*pns+2] = ptr_tinv[ii+3*pns+2]*sigma_mu;
				rQx[0] = ptr_lamt[ii+0*pns+2];
				rQx[1] = ptr_lamt[ii+1*pns+2];
				rqx[0] = ptr_lam[ii+0*pns+2] + ptr_dlam[ii+0*pns+2] + ptr_lamt[ii+0*pns+2]*ptr_db[ii+0*pns+2];
				rqx[1] = ptr_lam[ii+1*pns+2] + ptr_dlam[ii+1*pns+2] + ptr_lamt[ii+1*pns+2]*ptr_db[ii+1*pns+2];
				ptr_Zl[ii+0*pns+2] = 1.0 / (ptr_Z[ii+0*pns+2] + rQx[0] + ptr_lamt[ii+2*pns+2]);
				ptr_Zl[ii+1*pns+2] = 1.0 / (ptr_Z[ii+1*pns+2] + rQx[1] + ptr_lamt[ii+3*pns+2]);
				ptr_zl[ii+0*pns+2] = - ptr_z[ii+0*pns+2] + rqx[0] + ptr_lam[ii+2*pns+2] + ptr_dlam[ii+2*pns+2];
				ptr_zl[ii+1*pns+2] = - ptr_z[ii+1*pns+2] + rqx[1] + ptr_lam[ii+3*pns+2] + ptr_dlam[ii+3*pns+2];
				rqx[0] = rqx[0] - rQx[0]*ptr_zl[ii+0*pns+2]*ptr_Zl[ii+0*pns+2]; // update this before Qx !!!!!!!!!!!
				rqx[1] = rqx[1] - rQx[1]*ptr_zl[ii+1*pns+2]*ptr_Zl[ii+1*pns+2]; // update this before Qx !!!!!!!!!!!
				rQx[0] = rQx[0] - rQx[0]*rQx[0]*ptr_Zl[ii+0*pns+2];
				rQx[1] = rQx[1] - rQx[1]*rQx[1]*ptr_Zl[ii+1*pns+2];
				ptr_pd[nb0+ii+2] = ptr_bd[nb0+ii+2] + rQx[1] + rQx[0];
				ptr_pl[nb0+ii+2] = ptr_bl[nb0+ii+2] + rqx[1] - rqx[0];

				ptr_tinv[ii+0*pns+3] = 1.0/ptr_t[ii+0*pns+3];
				ptr_tinv[ii+1*pns+3] = 1.0/ptr_t[ii+1*pns+3];
				ptr_tinv[ii+2*pns+3] = 1.0/ptr_t[ii+2*pns+3];
				ptr_tinv[ii+3*pns+3] = 1.0/ptr_t[ii+3*pns+3];
				ptr_lamt[ii+0*pns+3] = ptr_lam[ii+0*pns+3]*ptr_tinv[ii+0*pns+3];
				ptr_lamt[ii+1*pns+3] = ptr_lam[ii+1*pns+3]*ptr_tinv[ii+1*pns+3];
				ptr_lamt[ii+2*pns+3] = ptr_lam[ii+2*pns+3]*ptr_tinv[ii+2*pns+3];
				ptr_lamt[ii+3*pns+3] = ptr_lam[ii+3*pns+3]*ptr_tinv[ii+3*pns+3];
				ptr_dlam[ii+0*pns+3] = ptr_tinv[ii+0*pns+3]*sigma_mu;
				ptr_dlam[ii+1*pns+3] = ptr_tinv[ii+1*pns+3]*sigma_mu;
				ptr_dlam[ii+2*pns+3] = ptr_tinv[ii+2*pns+3]*sigma_mu;
				ptr_dlam[ii+3*pns+3] = ptr_tinv[ii+3*pns+3]*sigma_mu;
				rQx[0] = ptr_lamt[ii+0*pns+3];
				rQx[1] = ptr_lamt[ii+1*pns+3];
				rqx[0] = ptr_lam[ii+0*pns+3] + ptr_dlam[ii+0*pns+3] + ptr_lamt[ii+0*pns+3]*ptr_db[ii+0*pns+3];
				rqx[1] = ptr_lam[ii+1*pns+3] + ptr_dlam[ii+1*pns+3] + ptr_lamt[ii+1*pns+3]*ptr_db[ii+1*pns+3];
				ptr_Zl[ii+0*pns+3] = 1.0 / (ptr_Z[ii+0*pns+3] + rQx[0] + ptr_lamt[ii+2*pns+3]);
				ptr_Zl[ii+1*pns+3] = 1.0 / (ptr_Z[ii+1*pns+3] + rQx[1] + ptr_lamt[ii+3*pns+3]);
				ptr_zl[ii+0*pns+3] = - ptr_z[ii+0*pns+3] + rqx[0] + ptr_lam[ii+2*pns+3] + ptr_dlam[ii+2*pns+3];
				ptr_zl[ii+1*pns+3] = - ptr_z[ii+1*pns+3] + rqx[1] + ptr_lam[ii+3*pns+3] + ptr_dlam[ii+3*pns+3];
				rqx[0] = rqx[0] - rQx[0]*ptr_zl[ii+0*pns+3]*ptr_Zl[ii+0*pns+3]; // update this before Qx !!!!!!!!!!!
				rqx[1] = rqx[1] - rQx[1]*ptr_zl[ii+1*pns+3]*ptr_Zl[ii+1*pns+3]; // update this before Qx !!!!!!!!!!!
				rQx[0] = rQx[0] - rQx[0]*rQx[0]*ptr_Zl[ii+0*pns+3];
				rQx[1] = rQx[1] - rQx[1]*rQx[1]*ptr_Zl[ii+1*pns+3];
				ptr_pd[nb0+ii+3] = ptr_bd[nb0+ii+3] + rQx[1] + rQx[0];
				ptr_pl[nb0+ii+3] = ptr_bl[nb0+ii+3] + rqx[1] - rqx[0];


				}
			for(; ii<ns0; ii++)
				{

				ptr_tinv[ii+0*pns+0] = 1.0/ptr_t[ii+0*pns+0];
				ptr_tinv[ii+1*pns+0] = 1.0/ptr_t[ii+1*pns+0];
				ptr_tinv[ii+2*pns+0] = 1.0/ptr_t[ii+2*pns+0];
				ptr_tinv[ii+3*pns+0] = 1.0/ptr_t[ii+3*pns+0];
				ptr_lamt[ii+0*pns+0] = ptr_lam[ii+0*pns+0]*ptr_tinv[ii+0*pns+0];
				ptr_lamt[ii+1*pns+0] = ptr_lam[ii+1*pns+0]*ptr_tinv[ii+1*pns+0];
				ptr_lamt[ii+2*pns+0] = ptr_lam[ii+2*pns+0]*ptr_tinv[ii+2*pns+0];
				ptr_lamt[ii+3*pns+0] = ptr_lam[ii+3*pns+0]*ptr_tinv[ii+3*pns+0];
				ptr_dlam[ii+0*pns+0] = ptr_tinv[ii+0*pns+0]*sigma_mu;
				ptr_dlam[ii+1*pns+0] = ptr_tinv[ii+1*pns+0]*sigma_mu;
				ptr_dlam[ii+2*pns+0] = ptr_tinv[ii+2*pns+0]*sigma_mu;
				ptr_dlam[ii+3*pns+0] = ptr_tinv[ii+3*pns+0]*sigma_mu;
				rQx[0] = ptr_lamt[ii+0*pns+0];
				rQx[1] = ptr_lamt[ii+1*pns+0];
				rqx[0] = ptr_lam[ii+0*pns+0] + ptr_dlam[ii+0*pns+0] + ptr_lamt[ii+0*pns+0]*ptr_db[ii+0*pns+0];
				rqx[1] = ptr_lam[ii+1*pns+0] + ptr_dlam[ii+1*pns+0] + ptr_lamt[ii+1*pns+0]*ptr_db[ii+1*pns+0];
				ptr_Zl[ii+0*pns+0] = 1.0 / (ptr_Z[ii+0*pns+0] + rQx[0] + ptr_lamt[ii+2*pns+0]);
				ptr_Zl[ii+1*pns+0] = 1.0 / (ptr_Z[ii+1*pns+0] + rQx[1] + ptr_lamt[ii+3*pns+0]);
				ptr_zl[ii+0*pns+0] = - ptr_z[ii+0*pns+0] + rqx[0] + ptr_lam[ii+2*pns+0] + ptr_dlam[ii+2*pns+0];
				ptr_zl[ii+1*pns+0] = - ptr_z[ii+1*pns+0] + rqx[1] + ptr_lam[ii+3*pns+0] + ptr_dlam[ii+3*pns+0];
				rqx[0] = rqx[0] - rQx[0]*ptr_zl[ii+0*pns+0]*ptr_Zl[ii+0*pns+0]; // update this before Qx !!!!!!!!!!!
				rqx[1] = rqx[1] - rQx[1]*ptr_zl[ii+1*pns+0]*ptr_Zl[ii+1*pns+0]; // update this before Qx !!!!!!!!!!!
				rQx[0] = rQx[0] - rQx[0]*rQx[0]*ptr_Zl[ii+0*pns+0];
				rQx[1] = rQx[1] - rQx[1]*rQx[1]*ptr_Zl[ii+1*pns+0];
				ptr_pd[nb0+ii+0] = ptr_bd[nb0+ii+0] + rQx[1] + rQx[0];
				ptr_pl[nb0+ii+0] = ptr_bl[nb0+ii+0] + rqx[1] - rqx[0];

				}

			}

		}

	}

void d_update_hessian_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **Qx, double **qx, double **bd, double **bl, double **pd, double **pl, double **db)
	{
	
	const int nbu = nu<nb ? nu : nb ;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	//const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box and soft constraints
	const int pnb  = bs*((nb+bs-1)/bs); // simd aligned number of box constraints
	const int png  = bs*((ng+bs-1)/bs); // simd aligned number of general constraints
	const int pngN = bs*((ngN+bs-1)/bs); // simd aligned number of general constraints at last stage

	//const int k0 = nbu;
	//const int k1 = (nu/bs)*bs;
	//const int kmax = nb;
	
	double temp0, temp1;
	
	double 
		*ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Qx, *ptr_qx,
		*ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv;
	
	int ii, jj, bs0;
	
	// first stage
	jj = 0;
	
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
	for(; ii<nbu-3; ii+=4)
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
	for(; ii<nbu; ii++)
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
		ptr_tinv  = t_inv[jj];
		ptr_db    = db[jj];
		ptr_bd    = bd[jj];
		ptr_bl    = bl[jj];
		ptr_pd    = pd[jj];
		ptr_pl    = pl[jj];

		ii = 0;
		for(; ii<nb-3; ii+=4)
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
		for(; ii<nb; ii++)
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
	ptr_tinv  = t_inv[jj];
	ptr_db    = db[jj];
	ptr_bd    = bd[jj];
	ptr_bl    = bl[jj];
	ptr_pd    = pd[jj];
	ptr_pl    = pl[jj];

	ii=nu;
	for(; ii<nb-3; ii+=4)
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
	for(; ii<nb; ii++)
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
			ptr_tinv = t_inv[jj];
			ptr_db    = db[jj];
			ptr_bd    = bd[jj];
			ptr_bl    = bl[jj];
			ptr_pd    = pd[jj];
			ptr_pl    = pl[jj];

			ptr_Qx    = Qx[jj];
			ptr_qx    = qx[jj];

			ii = 2*pnb;
			for(; ii<2*pnb+ng-3; ii+=4)
				{

				ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_tinv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
				ptr_lamt[ii+png+0] = ptr_lam[ii+png+0]*ptr_tinv[ii+png+0];
				ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+0] = ptr_tinv[ii+png+0]*sigma_mu; // !!!!!
				ptr_qx[ii+0] =  ptr_lam[ii+png+0] + ptr_lamt[ii+png+0]*ptr_db[ii+png+0] + ptr_dlam[ii+png+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0];
				ptr_Qx[ii+0] = sqrt(ptr_lamt[ii+0] + ptr_lamt[ii+png+0]);

				ptr_tinv[ii+1] = 1.0/ptr_t[ii+1];
				ptr_tinv[ii+png+1] = 1.0/ptr_t[ii+png+1];
				ptr_lamt[ii+1] = ptr_lam[ii+1]*ptr_tinv[ii+1];
				ptr_lamt[ii+png+1] = ptr_lam[ii+png+1]*ptr_tinv[ii+png+1];
				ptr_dlam[ii+1] = ptr_tinv[ii+1]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+1] = ptr_tinv[ii+png+1]*sigma_mu; // !!!!!
				ptr_qx[ii+1] =  ptr_lam[ii+png+1] + ptr_lamt[ii+png+1]*ptr_db[ii+png+1] + ptr_dlam[ii+png+1] - ptr_lam[ii+1] - ptr_lamt[ii+1]*ptr_db[ii+1] - ptr_dlam[ii+1];
				ptr_Qx[ii+1] = sqrt(ptr_lamt[ii+1] + ptr_lamt[ii+png+1]);

				ptr_tinv[ii+2] = 1.0/ptr_t[ii+2];
				ptr_tinv[ii+png+2] = 1.0/ptr_t[ii+png+2];
				ptr_lamt[ii+2] = ptr_lam[ii+2]*ptr_tinv[ii+2];
				ptr_lamt[ii+png+2] = ptr_lam[ii+png+2]*ptr_tinv[ii+png+2];
				ptr_dlam[ii+2] = ptr_tinv[ii+2]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+2] = ptr_tinv[ii+png+2]*sigma_mu; // !!!!!
				ptr_qx[ii+2] =  ptr_lam[ii+png+2] + ptr_lamt[ii+png+2]*ptr_db[ii+png+2] + ptr_dlam[ii+png+2] - ptr_lam[ii+2] - ptr_lamt[ii+2]*ptr_db[ii+2] - ptr_dlam[ii+2];
				ptr_Qx[ii+2] = sqrt(ptr_lamt[ii+2] + ptr_lamt[ii+png+2]);

				ptr_tinv[ii+3] = 1.0/ptr_t[ii+3];
				ptr_tinv[ii+png+3] = 1.0/ptr_t[ii+png+3];
				ptr_lamt[ii+3] = ptr_lam[ii+3]*ptr_tinv[ii+3];
				ptr_lamt[ii+png+3] = ptr_lam[ii+png+3]*ptr_tinv[ii+png+3];
				ptr_dlam[ii+3] = ptr_tinv[ii+3]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+3] = ptr_tinv[ii+png+3]*sigma_mu; // !!!!!
				ptr_qx[ii+3] =  ptr_lam[ii+png+3] + ptr_lamt[ii+png+3]*ptr_db[ii+png+3] + ptr_dlam[ii+png+3] - ptr_lam[ii+3] - ptr_lamt[ii+3]*ptr_db[ii+3] - ptr_dlam[ii+3];
				ptr_Qx[ii+3] = sqrt(ptr_lamt[ii+3] + ptr_lamt[ii+png+3]);

				}
			for(; ii<2*pnb+ng; ii++)
				{

				ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_tinv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
				ptr_lamt[ii+png+0] = ptr_lam[ii+png+0]*ptr_tinv[ii+png+0];
				ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
				ptr_dlam[ii+png+0] = ptr_tinv[ii+png+0]*sigma_mu; // !!!!!
				ptr_qx[ii+0] =  ptr_lam[ii+png+0] + ptr_lamt[ii+png+0]*ptr_db[ii+png+0] + ptr_dlam[ii+png+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0];
				ptr_Qx[ii+0] = sqrt(ptr_lamt[ii+0] + ptr_lamt[ii+png+0]);

				}
			}

		}
	if(ngN>0)
		{

		ptr_t     = t[N];
		ptr_lam   = lam[N];
		ptr_lamt  = lamt[N];
		ptr_dlam  = dlam[N];
		ptr_tinv = t_inv[N];
		ptr_db    = db[N];
		ptr_bd    = bd[N];
		ptr_bl    = bl[N];
		ptr_pd    = pd[N];
		ptr_pl    = pl[N];

		ptr_Qx    = Qx[N];
		ptr_qx    = qx[N];

		ii = 2*pnb;
		for(; ii<2*pnb+ngN-3; ii+=4)
			{

			ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
			ptr_tinv[ii+pngN+0] = 1.0/ptr_t[ii+pngN+0];
			ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
			ptr_lamt[ii+pngN+0] = ptr_lam[ii+pngN+0]*ptr_tinv[ii+pngN+0];
			ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
			ptr_dlam[ii+pngN+0] = ptr_tinv[ii+pngN+0]*sigma_mu; // !!!!!
			ptr_qx[ii+0] =  ptr_lam[ii+pngN+0] + ptr_lamt[ii+pngN+0]*ptr_db[ii+pngN+0] + ptr_dlam[ii+pngN+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0];
			ptr_Qx[ii+0] = sqrt(ptr_lamt[ii+0] + ptr_lamt[ii+pngN+0]);

			ptr_tinv[ii+1] = 1.0/ptr_t[ii+1];
			ptr_tinv[ii+pngN+1] = 1.0/ptr_t[ii+pngN+1];
			ptr_lamt[ii+1] = ptr_lam[ii+1]*ptr_tinv[ii+1];
			ptr_lamt[ii+pngN+1] = ptr_lam[ii+pngN+1]*ptr_tinv[ii+pngN+1];
			ptr_dlam[ii+1] = ptr_tinv[ii+1]*sigma_mu; // !!!!!
			ptr_dlam[ii+pngN+1] = ptr_tinv[ii+pngN+1]*sigma_mu; // !!!!!
			ptr_qx[ii+1] =  ptr_lam[ii+pngN+1] + ptr_lamt[ii+pngN+1]*ptr_db[ii+pngN+1] + ptr_dlam[ii+pngN+1] - ptr_lam[ii+1] - ptr_lamt[ii+1]*ptr_db[ii+1] - ptr_dlam[ii+1];
			ptr_Qx[ii+1] = sqrt(ptr_lamt[ii+1] + ptr_lamt[ii+pngN+1]);

			ptr_tinv[ii+2] = 1.0/ptr_t[ii+2];
			ptr_tinv[ii+pngN+2] = 1.0/ptr_t[ii+pngN+2];
			ptr_lamt[ii+2] = ptr_lam[ii+2]*ptr_tinv[ii+2];
			ptr_lamt[ii+pngN+2] = ptr_lam[ii+pngN+2]*ptr_tinv[ii+pngN+2];
			ptr_dlam[ii+2] = ptr_tinv[ii+2]*sigma_mu; // !!!!!
			ptr_dlam[ii+pngN+2] = ptr_tinv[ii+pngN+2]*sigma_mu; // !!!!!
			ptr_qx[ii+2] =  ptr_lam[ii+pngN+2] + ptr_lamt[ii+pngN+2]*ptr_db[ii+pngN+2] + ptr_dlam[ii+pngN+2] - ptr_lam[ii+2] - ptr_lamt[ii+2]*ptr_db[ii+2] - ptr_dlam[ii+2];
			ptr_Qx[ii+2] = sqrt(ptr_lamt[ii+2] + ptr_lamt[ii+pngN+2]);

			ptr_tinv[ii+3] = 1.0/ptr_t[ii+3];
			ptr_tinv[ii+pngN+3] = 1.0/ptr_t[ii+pngN+3];
			ptr_lamt[ii+3] = ptr_lam[ii+3]*ptr_tinv[ii+3];
			ptr_lamt[ii+pngN+3] = ptr_lam[ii+pngN+3]*ptr_tinv[ii+pngN+3];
			ptr_dlam[ii+3] = ptr_tinv[ii+3]*sigma_mu; // !!!!!
			ptr_dlam[ii+pngN+3] = ptr_tinv[ii+pngN+3]*sigma_mu; // !!!!!
			ptr_qx[ii+3] =  ptr_lam[ii+pngN+3] + ptr_lamt[ii+pngN+3]*ptr_db[ii+pngN+3] + ptr_dlam[ii+pngN+3] - ptr_lam[ii+3] - ptr_lamt[ii+3]*ptr_db[ii+3] - ptr_dlam[ii+3];
			ptr_Qx[ii+3] = sqrt(ptr_lamt[ii+3] + ptr_lamt[ii+pngN+3]);

			}
		for(; ii<2*pnb+ngN; ii++)
			{

			ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
			ptr_tinv[ii+pngN+0] = 1.0/ptr_t[ii+pngN+0];
			ptr_lamt[ii+0] = ptr_lam[ii+0]*ptr_tinv[ii+0];
			ptr_lamt[ii+pngN+0] = ptr_lam[ii+pngN+0]*ptr_tinv[ii+pngN+0];
			ptr_dlam[ii+0] = ptr_tinv[ii+0]*sigma_mu; // !!!!!
			ptr_dlam[ii+pngN+0] = ptr_tinv[ii+pngN+0]*sigma_mu; // !!!!!
			ptr_qx[ii+0] =  ptr_lam[ii+pngN+0] + ptr_lamt[ii+pngN+0]*ptr_db[ii+pngN+0] + ptr_dlam[ii+pngN+0] - ptr_lam[ii+0] - ptr_lamt[ii+0]*ptr_db[ii+0] - ptr_dlam[ii+0];
			ptr_Qx[ii+0] = sqrt(ptr_lamt[ii+0] + ptr_lamt[ii+pngN+0]);

			}
		}


	}



void d_update_hessian_soft_mpc(int N, int nx, int nu, int nh, int ns, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **db, double **Z, double **z, double **Zl, double **zl)
	{

	int nb = nh + ns;

	int nhu = nu<nh ? nu : nh ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	//const int k0 = nbu;
	//const int k1 = (nu/bs)*bs;
	//const int kmax = nb;
	
	
	double temp0, temp1;
	
	double *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv, *ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Z, *ptr_z, *ptr_Zl, *ptr_zl;

	static double Qx[8] = {};
	static double qx[8] = {};
	
	int ii, jj, bs0;
	
	// first stage
	jj = 0;
	
	ptr_t     = t[0];
	ptr_lam   = lam[0];
	ptr_lamt  = lamt[0];
	ptr_dlam  = dlam[0];
	ptr_tinv  = t_inv[0];
	ptr_pd    = pd[0];
	ptr_pl    = pl[0];
	ptr_bd    = bd[0];
	ptr_bl    = bl[0];
	ptr_db    = db[0];
	//ptr_Z     = Z[0];
	//ptr_z     = z[0];
	//ptr_Zl    = Zl[0];
	//ptr_zl    = zl[0];
	
	ii = 0;
	// hard constraints on u only
	for(; ii<nhu-3; ii+=4)
		{

		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_pd[ii+0] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
		ptr_pl[ii+0] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

		ptr_tinv[2] = 1.0/ptr_t[2];
		ptr_tinv[3] = 1.0/ptr_t[3];
		ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
		ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
		ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
		ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
		ptr_pd[ii+1] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
		ptr_pl[ii+1] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2] - ptr_dlam[2];

		ptr_tinv[4] = 1.0/ptr_t[4];
		ptr_tinv[5] = 1.0/ptr_t[5];
		ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
		ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
		ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
		ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
		ptr_pd[ii+2] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
		ptr_pl[ii+2] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[4] - ptr_dlam[4];

		ptr_tinv[6] = 1.0/ptr_t[6];
		ptr_tinv[7] = 1.0/ptr_t[7];
		ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
		ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
		ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
		ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
		ptr_pd[ii+3] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
		ptr_pl[ii+3] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[6] - ptr_dlam[6];

		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_tinv  += 8;
		ptr_db    += 8;

		}
	for(; ii<nhu; ii++)
		{
		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_pd[ii] = ptr_bd[ii] + ptr_lamt[0] + ptr_lamt[1];
		ptr_pl[ii] = ptr_bl[ii] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

		ptr_t     += 2;
		ptr_lam   += 2;
		ptr_lamt  += 2;
		ptr_dlam  += 2;
		ptr_tinv  += 2;
		ptr_db    += 2;
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
		ptr_tinv  = t_inv[jj];
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

			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_pd[ii+0] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
			ptr_pl[ii+0] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

			ptr_tinv[2] = 1.0/ptr_t[2];
			ptr_tinv[3] = 1.0/ptr_t[3];
			ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
			ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
			ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
			ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
			ptr_pd[ii+1] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
			ptr_pl[ii+1] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2] - ptr_dlam[2];

			ptr_tinv[4] = 1.0/ptr_t[4];
			ptr_tinv[5] = 1.0/ptr_t[5];
			ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
			ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
			ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
			ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
			ptr_pd[ii+2] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
			ptr_pl[ii+2] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[4] - ptr_dlam[4];

			ptr_tinv[6] = 1.0/ptr_t[6];
			ptr_tinv[7] = 1.0/ptr_t[7];
			ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
			ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
			ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
			ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
			ptr_pd[ii+3] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
			ptr_pl[ii+3] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[6] - ptr_dlam[6];

			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_tinv  += 8;
			ptr_db    += 8;
			ptr_Z     += 8;
			ptr_z     += 8;
			ptr_Zl    += 8;
			ptr_zl    += 8;

			}
		for(; ii<nh; ii++)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_pd[ii] = ptr_bd[ii] + ptr_lamt[0] + ptr_lamt[1];
			ptr_pl[ii] = ptr_bl[ii] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_tinv  += 2;
			ptr_db    += 2;
			ptr_Z     += 2;
			ptr_z     += 2;
			ptr_Zl    += 2;
			ptr_zl    += 2;
			}

		// soft constraints on states
		for(; ii<nb-3; ii+=4)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_tinv[pnb+0] = 1.0/ptr_t[pnb+0];
			ptr_tinv[pnb+1] = 1.0/ptr_t[pnb+1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_lamt[pnb+0] = ptr_lam[pnb+0]*ptr_tinv[pnb+0];
			ptr_lamt[pnb+1] = ptr_lam[pnb+1]*ptr_tinv[pnb+1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_dlam[pnb+0] = ptr_tinv[pnb+0]*sigma_mu; // !!!!!
			ptr_dlam[pnb+1] = ptr_tinv[pnb+1]*sigma_mu; // !!!!!
			Qx[0] = ptr_lamt[0];
			Qx[1] = ptr_lamt[1];
			qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
			qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
			ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[pnb+0]); // inverted of updated diagonal !!!
			ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[pnb+1]); // inverted of updated diagonal !!!
			ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[pnb+0] + ptr_dlam[pnb+0];
			ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[pnb+1] + ptr_dlam[pnb+1];
			qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
			qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
			Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
			Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
			ptr_pd[ii+0] = ptr_bd[ii+0] + Qx[1] + Qx[0];
			ptr_pl[ii+0] = ptr_bl[ii+0] + qx[1] - qx[0];

			ptr_tinv[2] = 1.0/ptr_t[2];
			ptr_tinv[3] = 1.0/ptr_t[3];
			ptr_tinv[pnb+2] = 1.0/ptr_t[pnb+2];
			ptr_tinv[pnb+3] = 1.0/ptr_t[pnb+3];
			ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
			ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
			ptr_lamt[pnb+2] = ptr_lam[pnb+2]*ptr_tinv[pnb+2];
			ptr_lamt[pnb+3] = ptr_lam[pnb+3]*ptr_tinv[pnb+3];
			ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
			ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
			ptr_dlam[pnb+2] = ptr_tinv[pnb+2]*sigma_mu; // !!!!!
			ptr_dlam[pnb+3] = ptr_tinv[pnb+3]*sigma_mu; // !!!!!
			Qx[2] = ptr_lamt[2];
			Qx[3] = ptr_lamt[3];
			qx[2] = ptr_lam[2] + ptr_dlam[2] + ptr_lamt[2]*ptr_db[2];
			qx[3] = ptr_lam[3] + ptr_dlam[3] + ptr_lamt[3]*ptr_db[3];
			ptr_Zl[2] = 1.0 / (ptr_Z[2] + Qx[2] + ptr_lamt[pnb+2]); // inverted of updated diagonal !!!
			ptr_Zl[3] = 1.0 / (ptr_Z[3] + Qx[3] + ptr_lamt[pnb+3]); // inverted of updated diagonal !!!
			ptr_zl[2] = - ptr_z[2] + qx[2] + ptr_lam[pnb+2] + ptr_dlam[pnb+2];
			ptr_zl[3] = - ptr_z[3] + qx[3] + ptr_lam[pnb+3] + ptr_dlam[pnb+3];
			qx[2] = qx[2] - Qx[2]*ptr_zl[2]*ptr_Zl[2]; // update this before Qx !!!!!!!!!!!
			qx[3] = qx[3] - Qx[3]*ptr_zl[3]*ptr_Zl[3]; // update this before Qx !!!!!!!!!!!
			Qx[2] = Qx[2] - Qx[2]*Qx[2]*ptr_Zl[2];
			Qx[3] = Qx[3] - Qx[3]*Qx[3]*ptr_Zl[3];
			ptr_pd[ii+1] = ptr_bd[ii+1] + Qx[3] + Qx[2];
			ptr_pl[ii+1] = ptr_bl[ii+1] + qx[3] - qx[2];

			ptr_tinv[4] = 1.0/ptr_t[4];
			ptr_tinv[5] = 1.0/ptr_t[5];
			ptr_tinv[pnb+4] = 1.0/ptr_t[pnb+4];
			ptr_tinv[pnb+5] = 1.0/ptr_t[pnb+5];
			ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
			ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
			ptr_lamt[pnb+4] = ptr_lam[pnb+4]*ptr_tinv[pnb+4];
			ptr_lamt[pnb+5] = ptr_lam[pnb+5]*ptr_tinv[pnb+5];
			ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
			ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
			ptr_dlam[pnb+4] = ptr_tinv[pnb+4]*sigma_mu; // !!!!!
			ptr_dlam[pnb+5] = ptr_tinv[pnb+5]*sigma_mu; // !!!!!
			Qx[4] = ptr_lamt[4];
			Qx[5] = ptr_lamt[5];
			qx[4] = ptr_lam[4] + ptr_dlam[4] + ptr_lamt[4]*ptr_db[4];
			qx[5] = ptr_lam[5] + ptr_dlam[5] + ptr_lamt[5]*ptr_db[5];
			ptr_Zl[4] = 1.0 / (ptr_Z[4] + Qx[4] + ptr_lamt[pnb+4]); // inverted of updated diagonal !!!
			ptr_Zl[5] = 1.0 / (ptr_Z[5] + Qx[5] + ptr_lamt[pnb+5]); // inverted of updated diagonal !!!
			ptr_zl[4] = - ptr_z[4] + qx[4] + ptr_lam[pnb+4] + ptr_dlam[pnb+4];
			ptr_zl[5] = - ptr_z[5] + qx[5] + ptr_lam[pnb+5] + ptr_dlam[pnb+5];
			qx[4] = qx[4] - Qx[4]*ptr_zl[4]*ptr_Zl[4]; // update this before Qx !!!!!!!!!!!
			qx[5] = qx[5] - Qx[5]*ptr_zl[5]*ptr_Zl[5]; // update this before Qx !!!!!!!!!!!
			Qx[4] = Qx[4] - Qx[4]*Qx[4]*ptr_Zl[4];
			Qx[5] = Qx[5] - Qx[5]*Qx[5]*ptr_Zl[5];
			ptr_pd[ii+2] = ptr_bd[ii+2] + Qx[5] + Qx[4];
			ptr_pl[ii+2] = ptr_bl[ii+2] + qx[5] - qx[4];

			ptr_tinv[6] = 1.0/ptr_t[6];
			ptr_tinv[7] = 1.0/ptr_t[7];
			ptr_tinv[pnb+6] = 1.0/ptr_t[pnb+6];
			ptr_tinv[pnb+7] = 1.0/ptr_t[pnb+7];
			ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
			ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
			ptr_lamt[pnb+6] = ptr_lam[pnb+6]*ptr_tinv[pnb+6];
			ptr_lamt[pnb+7] = ptr_lam[pnb+7]*ptr_tinv[pnb+7];
			ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
			ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
			ptr_dlam[pnb+6] = ptr_tinv[pnb+6]*sigma_mu; // !!!!!
			ptr_dlam[pnb+7] = ptr_tinv[pnb+7]*sigma_mu; // !!!!!
			Qx[6] = ptr_lamt[6];
			Qx[7] = ptr_lamt[7];
			qx[6] = ptr_lam[6] + ptr_dlam[6] + ptr_lamt[6]*ptr_db[6];
			qx[7] = ptr_lam[7] + ptr_dlam[7] + ptr_lamt[7]*ptr_db[7];
			ptr_Zl[6] = 1.0 / (ptr_Z[6] + Qx[6] + ptr_lamt[pnb+6]); // inverted of updated diagonal !!!
			ptr_Zl[7] = 1.0 / (ptr_Z[7] + Qx[7] + ptr_lamt[pnb+7]); // inverted of updated diagonal !!!
			ptr_zl[6] = - ptr_z[6] + qx[6] + ptr_lam[pnb+6] + ptr_dlam[pnb+6];
			ptr_zl[7] = - ptr_z[7] + qx[7] + ptr_lam[pnb+7] + ptr_dlam[pnb+7];
			qx[6] = qx[6] - Qx[6]*ptr_zl[6]*ptr_Zl[6]; // update this before Qx !!!!!!!!!!!
			qx[7] = qx[7] - Qx[7]*ptr_zl[7]*ptr_Zl[7]; // update this before Qx !!!!!!!!!!!
			Qx[6] = Qx[6] - Qx[6]*Qx[6]*ptr_Zl[6];
			Qx[7] = Qx[7] - Qx[7]*Qx[7]*ptr_Zl[7];
			ptr_pd[ii+3] = ptr_bd[ii+3] + Qx[7] + Qx[6];
			ptr_pl[ii+3] = ptr_bl[ii+3] + qx[7] - qx[6];

			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_tinv  += 8;
			ptr_db    += 8;
			ptr_Z     += 8;
			ptr_z     += 8;
			ptr_Zl    += 8;
			ptr_zl    += 8;

			}
		for(; ii<nb; ii++)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_tinv[pnb+0] = 1.0/ptr_t[pnb+0];
			ptr_tinv[pnb+1] = 1.0/ptr_t[pnb+1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_lamt[pnb+0] = ptr_lam[pnb+0]*ptr_tinv[pnb+0];
			ptr_lamt[pnb+1] = ptr_lam[pnb+1]*ptr_tinv[pnb+1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_dlam[pnb+0] = ptr_tinv[pnb+0]*sigma_mu; // !!!!!
			ptr_dlam[pnb+1] = ptr_tinv[pnb+1]*sigma_mu; // !!!!!
			Qx[0] = ptr_lamt[0];
			Qx[1] = ptr_lamt[1];
			qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
			qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
			ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[pnb+0]); // inverted of updated diagonal !!!
			ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[pnb+1]); // inverted of updated diagonal !!!
			ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[pnb+0] + ptr_dlam[pnb+0];
			ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[pnb+1] + ptr_dlam[pnb+1];
			qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
			qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
			Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
			Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
			ptr_pd[ii] = ptr_bd[ii] + Qx[1] + Qx[0];
			ptr_pl[ii] = ptr_bl[ii] + qx[1] - qx[0];

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_tinv  += 2;
			ptr_db    += 2;
			ptr_Z     += 2;
			ptr_z     += 2;
			ptr_Zl    += 2;
			ptr_zl    += 2;
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
	ptr_tinv  = t_inv[N] + 2*nu;
	ptr_db    = db[N]    + 2*nu;
	ptr_Z     = Z[N]     + 2*nu;
	ptr_z     = z[N]     + 2*nu;
	ptr_Zl    = Zl[N]    + 2*nu;
	ptr_zl    = zl[N]    + 2*nu;
	ptr_pd    = pd[N];
	ptr_pl    = pl[N];
	ptr_bd    = bd[N];
	ptr_bl    = bl[N];

	ii=nu;
	for(; ii<nh-3; ii+=4)
		{
		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_pd[ii+0] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
		ptr_pl[ii+0] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

		ptr_tinv[2] = 1.0/ptr_t[2];
		ptr_tinv[3] = 1.0/ptr_t[3];
		ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
		ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
		ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
		ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
		ptr_pd[ii+1] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
		ptr_pl[ii+1] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2] - ptr_dlam[2];

		ptr_tinv[4] = 1.0/ptr_t[4];
		ptr_tinv[5] = 1.0/ptr_t[5];
		ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
		ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
		ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
		ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
		ptr_pd[ii+2] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
		ptr_pl[ii+2] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[4] - ptr_dlam[4];

		ptr_tinv[6] = 1.0/ptr_t[6];
		ptr_tinv[7] = 1.0/ptr_t[7];
		ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
		ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
		ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
		ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
		ptr_pd[ii+3] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
		ptr_pl[ii+3] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[6] - ptr_dlam[6];

		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_tinv  += 8;
		ptr_db    += 8;
		ptr_Z     += 8;
		ptr_z     += 8;
		ptr_Zl    += 8;
		ptr_zl    += 8;

		}
	// cleanup hard constraints
	for(; ii<nh; ii++)
		{
		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_pd[ii] = ptr_bd[ii] + ptr_lamt[0] + ptr_lamt[1];
		ptr_pl[ii] = ptr_bl[ii] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

		ptr_t     += 2;
		ptr_lam   += 2;
		ptr_lamt  += 2;
		ptr_dlam  += 2;
		ptr_tinv  += 2;
		ptr_db    += 2;
		ptr_Z     += 2;
		ptr_z     += 2;
		ptr_Zl    += 2;
		ptr_zl    += 2;
		}

	// soft constraints main loop
	for(; ii<nb-3; ii+=4)
		{
		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_tinv[pnb+0] = 1.0/ptr_t[pnb+0];
		ptr_tinv[pnb+1] = 1.0/ptr_t[pnb+1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_lamt[pnb+0] = ptr_lam[pnb+0]*ptr_tinv[pnb+0];
		ptr_lamt[pnb+1] = ptr_lam[pnb+1]*ptr_tinv[pnb+1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_dlam[pnb+0] = ptr_tinv[pnb+0]*sigma_mu; // !!!!!
		ptr_dlam[pnb+1] = ptr_tinv[pnb+1]*sigma_mu; // !!!!!
		Qx[0] = ptr_lamt[0];
		Qx[1] = ptr_lamt[1];
		qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
		qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
		ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[pnb+0]); // inverted of updated diagonal !!!
		ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[pnb+1]); // inverted of updated diagonal !!!
		ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[pnb+0] + ptr_dlam[pnb+0];
		ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[pnb+1] + ptr_dlam[pnb+1];
		qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
		qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
		Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
		Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
		ptr_pd[ii+0] = ptr_bd[ii+0] + Qx[1] + Qx[0];
		ptr_pl[ii+0] = ptr_bl[ii+0] + qx[1] - qx[0];

		ptr_tinv[2] = 1.0/ptr_t[2];
		ptr_tinv[3] = 1.0/ptr_t[3];
		ptr_tinv[pnb+2] = 1.0/ptr_t[pnb+2];
		ptr_tinv[pnb+3] = 1.0/ptr_t[pnb+3];
		ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
		ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
		ptr_lamt[pnb+2] = ptr_lam[pnb+2]*ptr_tinv[pnb+2];
		ptr_lamt[pnb+3] = ptr_lam[pnb+3]*ptr_tinv[pnb+3];
		ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
		ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
		ptr_dlam[pnb+2] = ptr_tinv[pnb+2]*sigma_mu; // !!!!!
		ptr_dlam[pnb+3] = ptr_tinv[pnb+3]*sigma_mu; // !!!!!
		Qx[2] = ptr_lamt[2];
		Qx[3] = ptr_lamt[3];
		qx[2] = ptr_lam[2] + ptr_dlam[2] + ptr_lamt[2]*ptr_db[2];
		qx[3] = ptr_lam[3] + ptr_dlam[3] + ptr_lamt[3]*ptr_db[3];
		ptr_Zl[2] = 1.0 / (ptr_Z[2] + Qx[2] + ptr_lamt[pnb+2]); // inverted of updated diagonal !!!
		ptr_Zl[3] = 1.0 / (ptr_Z[3] + Qx[3] + ptr_lamt[pnb+3]); // inverted of updated diagonal !!!
		ptr_zl[2] = - ptr_z[2] + qx[2] + ptr_lam[pnb+2] + ptr_dlam[pnb+2];
		ptr_zl[3] = - ptr_z[3] + qx[3] + ptr_lam[pnb+3] + ptr_dlam[pnb+3];
		qx[2] = qx[2] - Qx[2]*ptr_zl[2]*ptr_Zl[2]; // update this before Qx !!!!!!!!!!!
		qx[3] = qx[3] - Qx[3]*ptr_zl[3]*ptr_Zl[3]; // update this before Qx !!!!!!!!!!!
		Qx[2] = Qx[2] - Qx[2]*Qx[2]*ptr_Zl[2];
		Qx[3] = Qx[3] - Qx[3]*Qx[3]*ptr_Zl[3];
		ptr_pd[ii+1] = ptr_bd[ii+1] + Qx[3] + Qx[2];
		ptr_pl[ii+1] = ptr_bl[ii+1] + qx[3] - qx[2];

		ptr_tinv[4] = 1.0/ptr_t[4];
		ptr_tinv[5] = 1.0/ptr_t[5];
		ptr_tinv[pnb+4] = 1.0/ptr_t[pnb+4];
		ptr_tinv[pnb+5] = 1.0/ptr_t[pnb+5];
		ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
		ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
		ptr_lamt[pnb+4] = ptr_lam[pnb+4]*ptr_tinv[pnb+4];
		ptr_lamt[pnb+5] = ptr_lam[pnb+5]*ptr_tinv[pnb+5];
		ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
		ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
		ptr_dlam[pnb+4] = ptr_tinv[pnb+4]*sigma_mu; // !!!!!
		ptr_dlam[pnb+5] = ptr_tinv[pnb+5]*sigma_mu; // !!!!!
		Qx[4] = ptr_lamt[4];
		Qx[5] = ptr_lamt[5];
		qx[4] = ptr_lam[4] + ptr_dlam[4] + ptr_lamt[4]*ptr_db[4];
		qx[5] = ptr_lam[5] + ptr_dlam[5] + ptr_lamt[5]*ptr_db[5];
		ptr_Zl[4] = 1.0 / (ptr_Z[4] + Qx[4] + ptr_lamt[pnb+4]); // inverted of updated diagonal !!!
		ptr_Zl[5] = 1.0 / (ptr_Z[5] + Qx[5] + ptr_lamt[pnb+5]); // inverted of updated diagonal !!!
		ptr_zl[4] = - ptr_z[4] + qx[4] + ptr_lam[pnb+4] + ptr_dlam[pnb+4];
		ptr_zl[5] = - ptr_z[5] + qx[5] + ptr_lam[pnb+5] + ptr_dlam[pnb+5];
		qx[4] = qx[4] - Qx[4]*ptr_zl[4]*ptr_Zl[4]; // update this before Qx !!!!!!!!!!!
		qx[5] = qx[5] - Qx[5]*ptr_zl[5]*ptr_Zl[5]; // update this before Qx !!!!!!!!!!!
		Qx[4] = Qx[4] - Qx[4]*Qx[4]*ptr_Zl[4];
		Qx[5] = Qx[5] - Qx[5]*Qx[5]*ptr_Zl[5];
		ptr_pd[ii+2] = ptr_bd[ii+2] + Qx[5] + Qx[4];
		ptr_pl[ii+2] = ptr_bl[ii+2] + qx[5] - qx[4];

		ptr_tinv[6] = 1.0/ptr_t[6];
		ptr_tinv[7] = 1.0/ptr_t[7];
		ptr_tinv[pnb+6] = 1.0/ptr_t[pnb+6];
		ptr_tinv[pnb+7] = 1.0/ptr_t[pnb+7];
		ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
		ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
		ptr_lamt[pnb+6] = ptr_lam[pnb+6]*ptr_tinv[pnb+6];
		ptr_lamt[pnb+7] = ptr_lam[pnb+7]*ptr_tinv[pnb+7];
		ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
		ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
		ptr_dlam[pnb+6] = ptr_tinv[pnb+6]*sigma_mu; // !!!!!
		ptr_dlam[pnb+7] = ptr_tinv[pnb+7]*sigma_mu; // !!!!!
		Qx[6] = ptr_lamt[6];
		Qx[7] = ptr_lamt[7];
		qx[6] = ptr_lam[6] + ptr_dlam[6] + ptr_lamt[6]*ptr_db[6];
		qx[7] = ptr_lam[7] + ptr_dlam[7] + ptr_lamt[7]*ptr_db[7];
		ptr_Zl[6] = 1.0 / (ptr_Z[6] + Qx[6] + ptr_lamt[pnb+6]); // inverted of updated diagonal !!!
		ptr_Zl[7] = 1.0 / (ptr_Z[7] + Qx[7] + ptr_lamt[pnb+7]); // inverted of updated diagonal !!!
		ptr_zl[6] = - ptr_z[6] + qx[6] + ptr_lam[pnb+6] + ptr_dlam[pnb+6];
		ptr_zl[7] = - ptr_z[7] + qx[7] + ptr_lam[pnb+7] + ptr_dlam[pnb+7];
		qx[6] = qx[6] - Qx[6]*ptr_zl[6]*ptr_Zl[6]; // update this before Qx !!!!!!!!!!!
		qx[7] = qx[7] - Qx[7]*ptr_zl[7]*ptr_Zl[7]; // update this before Qx !!!!!!!!!!!
		Qx[6] = Qx[6] - Qx[6]*Qx[6]*ptr_Zl[6];
		Qx[7] = Qx[7] - Qx[7]*Qx[7]*ptr_Zl[7];
		ptr_pd[ii+3] = ptr_bd[ii+3] + Qx[7] + Qx[6];
		ptr_pl[ii+3] = ptr_bl[ii+3] + qx[7] - qx[6];

		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_tinv  += 8;
		ptr_db    += 8;
		ptr_Z     += 8;
		ptr_z     += 8;
		ptr_Zl    += 8;
		ptr_zl    += 8;

		}
	for(; ii<nb; ii++)
		{
		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_tinv[pnb+0] = 1.0/ptr_t[pnb+0];
		ptr_tinv[pnb+1] = 1.0/ptr_t[pnb+1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_lamt[pnb+0] = ptr_lam[pnb+0]*ptr_tinv[pnb+0];
		ptr_lamt[pnb+1] = ptr_lam[pnb+1]*ptr_tinv[pnb+1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_dlam[pnb+0] = ptr_tinv[pnb+0]*sigma_mu; // !!!!!
		ptr_dlam[pnb+1] = ptr_tinv[pnb+1]*sigma_mu; // !!!!!
		Qx[0] = ptr_lamt[0];
		Qx[1] = ptr_lamt[1];
		qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
		qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
		ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[pnb+0]); // inverted of updated diagonal !!!
		ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[pnb+1]); // inverted of updated diagonal !!!
		ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[pnb+0] + ptr_dlam[pnb+0];
		ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[pnb+1] + ptr_dlam[pnb+1];
		qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
		qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
		Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
		Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
		ptr_pd[ii] = ptr_bd[ii] + Qx[1] + Qx[0];
		ptr_pl[ii] = ptr_bl[ii] + qx[1] - qx[0];

		ptr_t     += 2;
		ptr_lam   += 2;
		ptr_lamt  += 2;
		ptr_dlam  += 2;
		ptr_tinv  += 2;
		ptr_db    += 2;
		ptr_Z     += 2;
		ptr_z     += 2;
		ptr_Zl    += 2;
		ptr_zl    += 2;
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

	int ii, jj;

	int nb0, pnb, ng0, png;

	double
		*ptr_dlam, *ptr_t_inv, *ptr_dt, *ptr_pl2, *ptr_qx;

	for(jj=0; jj<=N; jj++)
		{

		ptr_dlam  = dlam[jj];
		ptr_dt    = dt[jj];
		ptr_t_inv = t_inv[jj];
		ptr_pl2   = pl2[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints

			for(ii=0; ii<nb0; ii++)
				{
				ptr_dlam[0*pnb+ii] = ptr_t_inv[0*pnb+ii]*(sigma_mu - ptr_dlam[0*pnb+ii]*ptr_dt[0*pnb+ii]);
				ptr_dlam[1*pnb+ii] = ptr_t_inv[1*pnb+ii]*(sigma_mu - ptr_dlam[1*pnb+ii]*ptr_dt[1*pnb+ii]);
				ptr_pl2[ii] += ptr_dlam[1*pnb+ii] - ptr_dlam[0*pnb+ii];
				}

			ptr_dlam  += 2*pnb;
			ptr_dt    += 2*pnb;
			ptr_t_inv += 2*pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			ptr_qx    = qx[jj];

			png  = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints

			for(ii=0; ii<ng0; ii++)
				{
				ptr_dlam[0*png+ii] = ptr_t_inv[0*png+ii]*(sigma_mu - ptr_dlam[0*png+ii]*ptr_dt[0*png+ii]);
				ptr_dlam[1*png+ii] = ptr_t_inv[1*png+ii]*(sigma_mu - ptr_dlam[1*png+ii]*ptr_dt[1*png+ii]);
				ptr_qx[ii] += ptr_dlam[1*png+ii] - ptr_dlam[0*png+ii];
				}

			}

		}

	}



void d_update_gradient_soft_mpc_tv(int N, int *nx, int *nu, int *nb, int *ng, int *ns, double sigma_mu, double **dt, double **dlam, double **t_inv, double **lamt, double **pl2, double **qxr, double **Zl, double **zl)
	{

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;

	int nb0, pnb, ng0, png, ns0, pns;

	double
		*ptr_dlam, *ptr_t_inv, *ptr_dt, *ptr_lamt, *ptr_pl2, *ptr_qx, *ptr_Zl, *ptr_zl;

	static double Qx[2] = {};
	static double qx[2] = {};

	for(jj=0; jj<=N; jj++)
		{

		ptr_dlam  = dlam[jj];
		ptr_dt    = dt[jj];
		ptr_lamt  = lamt[jj];
		ptr_t_inv = t_inv[jj];
		ptr_pl2   = pl2[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints

			for(ii=0; ii<nb0; ii++)
				{
				ptr_dlam[0*pnb+ii] = ptr_t_inv[0*pnb+ii]*(sigma_mu - ptr_dlam[0*pnb+ii]*ptr_dt[0*pnb+ii]);
				ptr_dlam[1*pnb+ii] = ptr_t_inv[1*pnb+ii]*(sigma_mu - ptr_dlam[1*pnb+ii]*ptr_dt[1*pnb+ii]);
				ptr_pl2[ii] += ptr_dlam[1*pnb+ii] - ptr_dlam[0*pnb+ii];
				}

			ptr_dlam  += 2*pnb;
			ptr_dt    += 2*pnb;
			ptr_lamt  += 2*pnb;
			ptr_t_inv += 2*pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			ptr_qx    = qxr[jj];

			png  = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints

			for(ii=2*pnb; ii<2*pnb+ng0; ii++)
				{
				ptr_dlam[0*png+ii] = ptr_t_inv[0*png+ii]*(sigma_mu - ptr_dlam[0*png+ii]*ptr_dt[0*png+ii]);
				ptr_dlam[1*png+ii] = ptr_t_inv[1*png+ii]*(sigma_mu - ptr_dlam[1*png+ii]*ptr_dt[1*png+ii]);
				ptr_qx[ii] += ptr_dlam[1*png+ii] - ptr_dlam[0*png+ii];
				}

			ptr_dlam  += 2*png;
			ptr_dt    += 2*png;
			ptr_lamt  += 2*png;
			ptr_t_inv += 2*png;

			}

		// box soft constraitns
		ns0 = ns[jj];
		if(ns0>0)
			{

			ptr_Zl    = Zl[jj];
			ptr_zl    = zl[jj];

			pns  = (ns0+bs-1)/bs*bs; // simd aligned number of box soft constraints

			for(ii=0; ii<ns0; ii++)
				{
				ptr_dlam[0*pns+ii] = ptr_t_inv[0*pns+ii]*(sigma_mu - ptr_dlam[0*pns+ii]*ptr_dt[0*pns+ii]);
				ptr_dlam[1*pns+ii] = ptr_t_inv[1*pns+ii]*(sigma_mu - ptr_dlam[1*pns+ii]*ptr_dt[1*pns+ii]);
				ptr_dlam[2*pns+ii] = ptr_t_inv[2*pns+ii]*(sigma_mu - ptr_dlam[2*pns+ii]*ptr_dt[2*pns+ii]);
				ptr_dlam[3*pns+ii] = ptr_t_inv[3*pns+ii]*(sigma_mu - ptr_dlam[3*pns+ii]*ptr_dt[3*pns+ii]);
				Qx[0] = ptr_lamt[0*pns+ii];
				Qx[1] = ptr_lamt[1*pns+ii];
				qx[0] = ptr_dlam[0*pns+ii];
				qx[1] = ptr_dlam[1*pns+ii];
				ptr_zl[0*pns+ii] += qx[0] + ptr_dlam[2*pns+ii];
				ptr_zl[1*pns+ii] += qx[1] + ptr_dlam[3*pns+ii];
				qx[0] = qx[0] - Qx[0]*(qx[0] + ptr_dlam[2*pns+ii])*ptr_Zl[0*pns+ii];
				qx[1] = qx[1] - Qx[1]*(qx[1] + ptr_dlam[3*pns+ii])*ptr_Zl[1*pns+ii];
				ptr_pl2[nb0+ii] += qx[1] - qx[0];
				}

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

	int nu0, nx0, nb0, pnb, ng0, png, cng;

	double alpha = ptr_alpha[0];
	
	double
		*ptr_db, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lamt, *ptr_lam, *ptr_dlam;
	
	int
		*ptr_idxb;
	
	int jj, ll;

	for(jj=0; jj<=N; jj++)
		{

		ptr_db   = db[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lamt = lamt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];
		ptr_idxb = idxb[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb = (nb0+bs-1)/bs*bs;

			// box constraints
			for(ll=0; ll<nb0; ll++)
				{

				ptr_dt[ll+0]   =   ptr_dux[ptr_idxb[ll]] - ptr_db[ll+0]   - ptr_t[ll+0];
				ptr_dt[ll+pnb] = - ptr_dux[ptr_idxb[ll]] - ptr_db[ll+pnb] - ptr_t[ll+pnb];
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

			ptr_db   += 2*pnb;
			ptr_dux  += 2*pnb;
			ptr_t    += 2*pnb;
			ptr_dt   += 2*pnb;
			ptr_lamt += 2*pnb;
			ptr_lam  += 2*pnb;
			ptr_dlam += 2*pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			nu0 = nu[jj];
			nx0 = nx[jj];
			png = (ng0+bs-1)/bs*bs;
			cng = (ng0+ncl-1)/ncl*ncl;

			dgemv_t_lib(nx0+nu0, ng0, pDCt[jj], cng, ptr_dux, 0, ptr_dt, ptr_dt);

			for(ll=0; ll<ng0; ll++)
				{
				ptr_dt[ll+png] = - ptr_dt[ll];
				ptr_dt[ll+0]   += - ptr_db[ll+0]   - ptr_t[ll+0];
				ptr_dt[ll+png] += - ptr_db[ll+png] - ptr_t[ll+png];
				ptr_dlam[ll+0]   -= ptr_lamt[ll+0]   * ptr_dt[ll+0]   + ptr_lam[ll+0];
				ptr_dlam[ll+png] -= ptr_lamt[ll+png] * ptr_dt[ll+png] + ptr_lam[ll+png];
				if( -alpha*ptr_dlam[ll+0]>ptr_lam[ll+0] )
					{
					alpha = - ptr_lam[ll+0] / ptr_dlam[ll+0];
					}
				if( -alpha*ptr_dlam[ll+png]>ptr_lam[ll+png] )
					{
					alpha = - ptr_lam[ll+png] / ptr_dlam[ll+png];
					}
				if( -alpha*ptr_dt[ll+0]>ptr_t[ll+0] )
					{
					alpha = - ptr_t[ll+0] / ptr_dt[ll+0];
					}
				if( -alpha*ptr_dt[ll+png]>ptr_t[ll+png] )
					{
					alpha = - ptr_t[ll+png] / ptr_dt[ll+png];
					}

				}

			}

		}		

	// store alpha
	ptr_alpha[0] = alpha;

	return;
	
	}



void d_compute_alpha_soft_mpc_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, int *ns, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **pDCt, double **db, double **Zl, double **zl)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nu0, nx0, nb0, pnb, ng0, png, cng, ns0, pns;

	double alpha = ptr_alpha[0];
	
	double
		*ptr_db, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lamt, *ptr_lam, *ptr_dlam, *ptr_zl, *ptr_Zl;
	
	int
		*ptr_idxb;
	
	int jj, ll;

	for(jj=0; jj<=N; jj++)
		{

		ptr_db   = db[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lamt = lamt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];
		ptr_idxb = idxb[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb = (nb0+bs-1)/bs*bs;

			// box constraints
			for(ll=0; ll<nb0; ll++)
				{

				ptr_dt[ll+0]   =   ptr_dux[ptr_idxb[ll]] - ptr_db[ll+0]   - ptr_t[ll+0];
				ptr_dt[ll+pnb] = - ptr_dux[ptr_idxb[ll]] - ptr_db[ll+pnb] - ptr_t[ll+pnb];
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

			ptr_db   += 2*pnb;
			ptr_t    += 2*pnb;
			ptr_dt   += 2*pnb;
			ptr_lamt += 2*pnb;
			ptr_lam  += 2*pnb;
			ptr_dlam += 2*pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			nu0 = nu[jj];
			nx0 = nx[jj];
			png = (ng0+bs-1)/bs*bs;
			cng = (ng0+ncl-1)/ncl*ncl;

			dgemv_t_lib(nx0+nu0, ng0, pDCt[jj], cng, ptr_dux, 0, ptr_dt, ptr_dt);

			for(ll=0; ll<ng0; ll++)
				{
				ptr_dt[ll+png] = - ptr_dt[ll];
				ptr_dt[ll+0]   += - ptr_db[ll+0]   - ptr_t[ll+0];
				ptr_dt[ll+png] += - ptr_db[ll+png] - ptr_t[ll+png];
				ptr_dlam[ll+0]   -= ptr_lamt[ll+0]   * ptr_dt[ll+0]   + ptr_lam[ll+0];
				ptr_dlam[ll+png] -= ptr_lamt[ll+png] * ptr_dt[ll+png] + ptr_lam[ll+png];
				if( -alpha*ptr_dlam[ll+0]>ptr_lam[ll+0] )
					{
					alpha = - ptr_lam[ll+0] / ptr_dlam[ll+0];
					}
				if( -alpha*ptr_dlam[ll+png]>ptr_lam[ll+png] )
					{
					alpha = - ptr_lam[ll+png] / ptr_dlam[ll+png];
					}
				if( -alpha*ptr_dt[ll+0]>ptr_t[ll+0] )
					{
					alpha = - ptr_t[ll+0] / ptr_dt[ll+0];
					}
				if( -alpha*ptr_dt[ll+png]>ptr_t[ll+png] )
					{
					alpha = - ptr_t[ll+png] / ptr_dt[ll+png];
					}

				}

			ptr_db   += 2*png;
			ptr_t    += 2*png;
			ptr_dt   += 2*png;
			ptr_lamt += 2*png;
			ptr_lam  += 2*png;
			ptr_dlam += 2*png;

			}

		// box soft constraints
		ns0 = ns[jj];
		if(ns0>0)
			{

			ptr_Zl   = Zl[jj];
			ptr_zl   = zl[jj];

			pns = (ns0+bs-1)/bs*bs;

			// box constraints
			for(ll=0; ll<ns0; ll++)
				{
				ptr_dt[2*pns+ll] = ( ptr_zl[0*pns+ll] - ptr_lamt[0*pns+ll]*ptr_dux[ptr_idxb[nb0+ll]] ) * ptr_Zl[0*pns+ll];
				ptr_dt[3*pns+ll] = ( ptr_zl[1*pns+ll] + ptr_lamt[1*pns+ll]*ptr_dux[ptr_idxb[nb0+ll]] ) * ptr_Zl[1*pns+ll];
				ptr_dt[0*pns+ll] = ptr_dt[2*pns+ll] + ptr_dux[ptr_idxb[nb0+ll]] - ptr_db[0*pns+ll] - ptr_t[0*pns+ll];
				ptr_dt[1*pns+ll] = ptr_dt[3*pns+ll] - ptr_dux[ptr_idxb[nb0+ll]] - ptr_db[1*pns+ll] - ptr_t[1*pns+ll];
				ptr_dt[2*pns+ll] -= ptr_t[2*pns+ll];
				ptr_dt[3*pns+ll] -= ptr_t[3*pns+ll];
				ptr_dlam[0*pns+ll] -= ptr_lamt[0*pns+ll] * ptr_dt[0*pns+ll] + ptr_lam[0*pns+ll];
				ptr_dlam[1*pns+ll] -= ptr_lamt[1*pns+ll] * ptr_dt[1*pns+ll] + ptr_lam[1*pns+ll];
				ptr_dlam[2*pns+ll] -= ptr_lamt[2*pns+ll] * ptr_dt[2*pns+ll] + ptr_lam[2*pns+ll];
				ptr_dlam[3*pns+ll] -= ptr_lamt[3*pns+ll] * ptr_dt[3*pns+ll] + ptr_lam[3*pns+ll];
				if( -alpha*ptr_dlam[0*pns+ll]>ptr_lam[0*pns+ll] )
					{
					alpha = - ptr_lam[0*pns+ll] / ptr_dlam[0*pns+ll];
					}
				if( -alpha*ptr_dlam[1*pns+ll]>ptr_lam[1*pns+ll] )
					{
					alpha = - ptr_lam[1*pns+ll] / ptr_dlam[1*pns+ll];
					}
				if( -alpha*ptr_dlam[2*pns+ll]>ptr_lam[2*pns+ll] )
					{
					alpha = - ptr_lam[2*pns+ll] / ptr_dlam[2*pns+ll];
					}
				if( -alpha*ptr_dlam[3*pns+ll]>ptr_lam[3*pns+ll] )
					{
					alpha = - ptr_lam[3*pns+ll] / ptr_dlam[3*pns+ll];
					}
				if( -alpha*ptr_dt[0*pns+ll]>ptr_t[0*pns+ll] )
					{
					alpha = - ptr_t[0*pns+ll] / ptr_dt[0*pns+ll];
					}
				if( -alpha*ptr_dt[1*pns+ll]>ptr_t[1*pns+ll] )
					{
					alpha = - ptr_t[1*pns+ll] / ptr_dt[1*pns+ll];
					}
				if( -alpha*ptr_dt[2*pns+ll]>ptr_t[2*pns+ll] )
					{
					alpha = - ptr_t[2*pns+ll] / ptr_dt[2*pns+ll];
					}
				if( -alpha*ptr_dt[3*pns+ll]>ptr_t[3*pns+ll] )
					{
					alpha = - ptr_t[3*pns+ll] / ptr_dt[3*pns+ll];
					}

				}

			}

		}		
	
	// store alpha
	ptr_alpha[0] = alpha;

	return;
	
	}


void d_compute_alpha_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **pDCt, double **db)
	{
	
/*	const int bs = 4; //d_get_mr();*/

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

	double alpha = ptr_alpha[0];
	
/*	int kna = ((k1+bs-1)/bs)*bs;*/

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
	for(; ll<nbu; ll++)
		{

		ptr_dt[ll+0]   =   ptr_dux[ll] - ptr_db[ll+0]   - ptr_t[ll+0];
		ptr_dt[ll+pnb] = - ptr_dux[ll] - ptr_db[ll+pnb] - ptr_t[ll+pnb];
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
		for(; ll<nb; ll++)
			{

			ptr_dt[ll+0]   =   ptr_dux[ll] - ptr_db[ll+0]   - ptr_t[ll+0];
			ptr_dt[ll+pnb] = - ptr_dux[ll] - ptr_db[ll+pnb] - ptr_t[ll+pnb];
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
	for(; ll<nb; ll++)
		{

		ptr_dt[ll+0]   =   ptr_dux[ll] - ptr_db[ll+0]   - ptr_t[ll+0];
		ptr_dt[ll+pnb] = - ptr_dux[ll] - ptr_db[ll+pnb] - ptr_t[ll+pnb];
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

			for(ll=2*pnb; ll<2*pnb+ng; ll++)
				{
				ptr_dt[ll+png] = - ptr_dt[ll];
				ptr_dt[ll+0]   += - ptr_db[ll+0]   - ptr_t[ll+0];
				ptr_dt[ll+png] += - ptr_db[ll+png] - ptr_t[ll+png];
				ptr_dlam[ll+0]   -= ptr_lamt[ll+0]   * ptr_dt[ll+0]   + ptr_lam[ll+0];
				ptr_dlam[ll+png] -= ptr_lamt[ll+png] * ptr_dt[ll+png] + ptr_lam[ll+png];
				if( -alpha*ptr_dlam[ll+0]>ptr_lam[ll+0] )
					{
					alpha = - ptr_lam[ll+0] / ptr_dlam[ll+0];
					}
				if( -alpha*ptr_dlam[ll+png]>ptr_lam[ll+png] )
					{
					alpha = - ptr_lam[ll+png] / ptr_dlam[ll+png];
					}
				if( -alpha*ptr_dt[ll+0]>ptr_t[ll+0] )
					{
					alpha = - ptr_t[ll+0] / ptr_dt[ll+0];
					}
				if( -alpha*ptr_dt[ll+png]>ptr_t[ll+png] )
					{
					alpha = - ptr_t[ll+png] / ptr_dt[ll+png];
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

		for(ll=2*pnb; ll<2*pnb+ngN; ll++)
			{
			ptr_dt[ll+pngN] = - ptr_dt[ll];
			ptr_dt[ll+0]    += - ptr_db[ll+0]   - ptr_t[ll+0];
			ptr_dt[ll+pngN] += - ptr_db[ll+pngN] - ptr_t[ll+pngN];
			ptr_dlam[ll+0]    -= ptr_lamt[ll+0]   * ptr_dt[ll+0]   + ptr_lam[ll+0];
			ptr_dlam[ll+pngN] -= ptr_lamt[ll+pngN] * ptr_dt[ll+pngN] + ptr_lam[ll+pngN];
			if( -alpha*ptr_dlam[ll+0]>ptr_lam[ll+0] )
				{
				alpha = - ptr_lam[ll+0] / ptr_dlam[ll+0];
				}
			if( -alpha*ptr_dlam[ll+pngN]>ptr_lam[ll+pngN] )
				{
				alpha = - ptr_lam[ll+pngN] / ptr_dlam[ll+pngN];
				}
			if( -alpha*ptr_dt[ll+0]>ptr_t[ll+0] )
				{
				alpha = - ptr_t[ll+0] / ptr_dt[ll+0];
				}
			if( -alpha*ptr_dt[ll+pngN]>ptr_t[ll+pngN] )
				{
				alpha = - ptr_t[ll+pngN] / ptr_dt[ll+pngN];
				}

			}

		}
			

	// store alpha
	ptr_alpha[0] = alpha;

	return;
	
	}



void d_compute_alpha_soft_mpc(int N, int nx, int nu, int nh, int ns, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db, double **Zl, double **zl)
	{
	
/*	const int bs = 4; //d_get_mr();*/

	int nb = nh + ns;

	int nhu = nu<nh ? nu : nh ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	
	double alpha = ptr_alpha[0];
	
/*	int kna = ((k1+bs-1)/bs)*bs;*/

	int jj, ll;


	// first stage

	ll = 0;
	// hard input constraints
	for(; ll<nhu; ll++)
		{

		dt[0][2*ll+0] =   dux[0][ll] - db[0][2*ll+0] - t[0][2*ll+0];
		dt[0][2*ll+1] = - dux[0][ll] - db[0][2*ll+1] - t[0][2*ll+1];
		dlam[0][2*ll+0] -= lamt[0][2*ll+0] * dt[0][2*ll+0] + lam[0][2*ll+0];
		dlam[0][2*ll+1] -= lamt[0][2*ll+1] * dt[0][2*ll+1] + lam[0][2*ll+1];
		if( -alpha*dlam[0][2*ll+0]>lam[0][2*ll+0] )
			{
			alpha = - lam[0][2*ll+0] / dlam[0][2*ll+0];
			}
		if( -alpha*dlam[0][2*ll+1]>lam[0][2*ll+1] )
			{
			alpha = - lam[0][2*ll+1] / dlam[0][2*ll+1];
			}
		if( -alpha*dt[0][2*ll+0]>t[0][2*ll+0] )
			{
			alpha = - t[0][2*ll+0] / dt[0][2*ll+0];
			}
		if( -alpha*dt[0][2*ll+1]>t[0][2*ll+1] )
			{
			alpha = - t[0][2*ll+1] / dt[0][2*ll+1];
			}

		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{

		ll = 0;
		// hard input and state constraints
		for(; ll<nh; ll++)
			{

			dt[jj][2*ll+0] =   dux[jj][ll] - db[jj][2*ll+0] - t[jj][2*ll+0];
			dt[jj][2*ll+1] = - dux[jj][ll] - db[jj][2*ll+1] - t[jj][2*ll+1];
			dlam[jj][2*ll+0] -= lamt[jj][2*ll+0] * dt[jj][2*ll+0] + lam[jj][2*ll+0];
			dlam[jj][2*ll+1] -= lamt[jj][2*ll+1] * dt[jj][2*ll+1] + lam[jj][2*ll+1];
			if( -alpha*dlam[jj][2*ll+0]>lam[jj][2*ll+0] )
				{
				alpha = - lam[jj][2*ll+0] / dlam[jj][2*ll+0];
				}
			if( -alpha*dlam[jj][2*ll+1]>lam[jj][2*ll+1] )
				{
				alpha = - lam[jj][2*ll+1] / dlam[jj][2*ll+1];
				}
			if( -alpha*dt[jj][2*ll+0]>t[jj][2*ll+0] )
				{
				alpha = - t[jj][2*ll+0] / dt[jj][2*ll+0];
				}
			if( -alpha*dt[jj][2*ll+1]>t[jj][2*ll+1] )
				{
				alpha = - t[jj][2*ll+1] / dt[jj][2*ll+1];
				}

			}
		// soft state constraints
		for(; ll<nb; ll++)
			{

			dt[jj][pnb+2*ll+0] = ( zl[jj][2*ll+0] - lamt[jj][2*ll+0]*dux[jj][ll] ) * Zl[jj][2*ll+0];
			dt[jj][pnb+2*ll+1] = ( zl[jj][2*ll+1] + lamt[jj][2*ll+1]*dux[jj][ll] ) * Zl[jj][2*ll+1];
			dt[jj][2*ll+0] = dt[jj][pnb+2*ll+0] + dux[jj][ll] - db[jj][2*ll+0] - t[jj][2*ll+0];
			dt[jj][2*ll+1] = dt[jj][pnb+2*ll+1] - dux[jj][ll] - db[jj][2*ll+1] - t[jj][2*ll+1];
			dt[jj][pnb+2*ll+0] -= t[jj][pnb+2*ll+0];
			dt[jj][pnb+2*ll+1] -= t[jj][pnb+2*ll+1];
			dlam[jj][2*ll+0] -= lamt[jj][2*ll+0] * dt[jj][2*ll+0] + lam[jj][2*ll+0];
			dlam[jj][2*ll+1] -= lamt[jj][2*ll+1] * dt[jj][2*ll+1] + lam[jj][2*ll+1];
			dlam[jj][pnb+2*ll+0] -= lamt[jj][pnb+2*ll+0] * dt[jj][pnb+2*ll+0] + lam[jj][pnb+2*ll+0];
			dlam[jj][pnb+2*ll+1] -= lamt[jj][pnb+2*ll+1] * dt[jj][pnb+2*ll+1] + lam[jj][pnb+2*ll+1];
			if( -alpha*dlam[jj][2*ll+0]>lam[jj][2*ll+0] )
				{
				alpha = - lam[jj][2*ll+0] / dlam[jj][2*ll+0];
				}
			if( -alpha*dlam[jj][2*ll+1]>lam[jj][2*ll+1] )
				{
				alpha = - lam[jj][2*ll+1] / dlam[jj][2*ll+1];
				}
			if( -alpha*dt[jj][2*ll+0]>t[jj][2*ll+0] )
				{
				alpha = - t[jj][2*ll+0] / dt[jj][2*ll+0];
				}
			if( -alpha*dt[jj][2*ll+1]>t[jj][2*ll+1] )
				{
				alpha = - t[jj][2*ll+1] / dt[jj][2*ll+1];
				}
			if( -alpha*dlam[jj][pnb+2*ll+0]>lam[jj][pnb+2*ll+0] )
				{
				alpha = - lam[jj][pnb+2*ll+0] / dlam[jj][pnb+2*ll+0];
				}
			if( -alpha*dlam[jj][pnb+2*ll+1]>lam[jj][pnb+2*ll+1] )
				{
				alpha = - lam[jj][pnb+2*ll+1] / dlam[jj][pnb+2*ll+1];
				}
			if( -alpha*dt[jj][pnb+2*ll+0]>t[jj][pnb+2*ll+0] )
				{
				alpha = - t[jj][pnb+2*ll+0] / dt[jj][pnb+2*ll+0];
				}
			if( -alpha*dt[jj][pnb+2*ll+1]>t[jj][pnb+2*ll+1] )
				{
				alpha = - t[jj][pnb+2*ll+1] / dt[jj][pnb+2*ll+1];
				}

			}

		}		

	// last stage
	jj = N;
	ll = nu;
	// hard state constraints
	for(; ll<nh; ll++)
		{

		dt[jj][2*ll+0] =   dux[jj][ll] - db[jj][2*ll+0] - t[jj][2*ll+0];
		dt[jj][2*ll+1] = - dux[jj][ll] - db[jj][2*ll+1] - t[jj][2*ll+1];
		dlam[jj][2*ll+0] -= lamt[jj][2*ll+0] * dt[jj][2*ll+0] + lam[jj][2*ll+0];
		dlam[jj][2*ll+1] -= lamt[jj][2*ll+1] * dt[jj][2*ll+1] + lam[jj][2*ll+1];
		if( -alpha*dlam[jj][2*ll+0]>lam[jj][2*ll+0] )
			{
			alpha = - lam[jj][2*ll+0] / dlam[jj][2*ll+0];
			}
		if( -alpha*dlam[jj][2*ll+1]>lam[jj][2*ll+1] )
			{
			alpha = - lam[jj][2*ll+1] / dlam[jj][2*ll+1];
			}
		if( -alpha*dt[jj][2*ll+0]>t[jj][2*ll+0] )
			{
			alpha = - t[jj][2*ll+0] / dt[jj][2*ll+0];
			}
		if( -alpha*dt[jj][2*ll+1]>t[jj][2*ll+1] )
			{
			alpha = - t[jj][2*ll+1] / dt[jj][2*ll+1];
			}

		}
	// soft state constraints
	for(; ll<nb; ll++)
		{

		dt[N][pnb+2*ll+0] = ( zl[N][2*ll+0] - lamt[N][2*ll+0]*dux[N][ll] ) * Zl[N][2*ll+0];
		dt[N][pnb+2*ll+1] = ( zl[N][2*ll+1] + lamt[N][2*ll+1]*dux[N][ll] ) * Zl[N][2*ll+1];
		dt[N][2*ll+0] = dt[N][pnb+2*ll+0] + dux[N][ll] - db[N][2*ll+0] - t[N][2*ll+0];
		dt[N][2*ll+1] = dt[N][pnb+2*ll+1] - dux[N][ll] - db[N][2*ll+1] - t[N][2*ll+1];
		dt[N][pnb+2*ll+0] -= t[N][pnb+2*ll+0];
		dt[N][pnb+2*ll+1] -= t[N][pnb+2*ll+1];
		dlam[N][2*ll+0] -= lamt[N][2*ll+0] * dt[N][2*ll+0] + lam[N][2*ll+0];
		dlam[N][2*ll+1] -= lamt[N][2*ll+1] * dt[N][2*ll+1] + lam[N][2*ll+1];
		dlam[N][pnb+2*ll+0] -= lamt[N][pnb+2*ll+0] * dt[N][pnb+2*ll+0] + lam[N][pnb+2*ll+0];
		dlam[N][pnb+2*ll+1] -= lamt[N][pnb+2*ll+1] * dt[N][pnb+2*ll+1] + lam[N][pnb+2*ll+1];
		if( -alpha*dlam[N][2*ll+0]>lam[N][2*ll+0] )
			{
			alpha = - lam[N][2*ll+0] / dlam[N][2*ll+0];
			}
		if( -alpha*dlam[N][2*ll+1]>lam[N][2*ll+1] )
			{
			alpha = - lam[N][2*ll+1] / dlam[N][2*ll+1];
			}
		if( -alpha*dt[N][2*ll+0]>t[N][2*ll+0] )
			{
			alpha = - t[N][2*ll+0] / dt[N][2*ll+0];
			}
		if( -alpha*dt[N][2*ll+1]>t[N][2*ll+1] )
			{
			alpha = - t[N][2*ll+1] / dt[N][2*ll+1];
			}
		if( -alpha*dlam[N][pnb+2*ll+0]>lam[N][pnb+2*ll+0] )
			{
			alpha = - lam[N][pnb+2*ll+0] / dlam[N][pnb+2*ll+0];
			}
		if( -alpha*dlam[N][pnb+2*ll+1]>lam[N][pnb+2*ll+1] )
			{
			alpha = - lam[N][pnb+2*ll+1] / dlam[N][pnb+2*ll+1];
			}
		if( -alpha*dt[N][pnb+2*ll+0]>t[N][pnb+2*ll+0] )
			{
			alpha = - t[N][pnb+2*ll+0] / dt[N][pnb+2*ll+0];
			}
		if( -alpha*dt[N][pnb+2*ll+1]>t[N][pnb+2*ll+1] )
			{
			alpha = - t[N][pnb+2*ll+1] / dt[N][pnb+2*ll+1];
			}


		}
	
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
	
	int
		*ptr_idxb;
	
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

		ptr_idxb = idxb[jj];

		ll = 0;
		for(; ll<nb[jj]; ll++)
			{

			ptr_dt[ll+0]   =   ptr_dux[ptr_idxb[ll]] - ptr_db[ll+0]   - ptr_t[ll+0];
			ptr_dt[ll+pnb] = - ptr_dux[ptr_idxb[ll]] - ptr_db[ll+pnb] - ptr_t[ll+pnb];
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

	int nu0, nx0, nb0, pnb, ng0, png;

	int jj, ll;
	
	double
		*ptr_pi, *ptr_dpi, *ptr_ux, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lam, *ptr_dlam;

	double mu = 0;

	for(jj=0; jj<=N; jj++)
		{

		nx0 = nx[jj];
		nu0 = nu[jj];
		nb0 = nb[jj];
		pnb = bs*((nb0+bs-1)/bs); // cache aligned number of box constraints
		ng0 = ng[jj];
		png = bs*((ng0+bs-1)/bs); // cache aligned number of box constraints
		
		ptr_pi   = pi[jj];
		ptr_dpi  = dpi[jj];
		ptr_ux   = ux[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

		// update inputs and states
		for(ll=0; ll<nu0+nx0-3; ll+=4)
			{
			ptr_ux[ll+0] += alpha*(ptr_dux[ll+0] - ptr_ux[ll+0]);
			ptr_ux[ll+1] += alpha*(ptr_dux[ll+1] - ptr_ux[ll+1]);
			ptr_ux[ll+2] += alpha*(ptr_dux[ll+2] - ptr_ux[ll+2]);
			ptr_ux[ll+3] += alpha*(ptr_dux[ll+3] - ptr_ux[ll+3]);
			}
		for(; ll<nu0+nx0; ll++)
			ptr_ux[ll] += alpha*(ptr_dux[ll] - ptr_ux[ll]);
		// update equality constrained multipliers
		for(ll=0; ll<nx0-3; ll+=4)
			{
			ptr_pi[ll+0] += alpha*(ptr_dpi[ll+0] - ptr_pi[ll+0]);
			ptr_pi[ll+1] += alpha*(ptr_dpi[ll+1] - ptr_pi[ll+1]);
			ptr_pi[ll+2] += alpha*(ptr_dpi[ll+2] - ptr_pi[ll+2]);
			ptr_pi[ll+3] += alpha*(ptr_dpi[ll+3] - ptr_pi[ll+3]);
			}
		for(; ll<nx0; ll++)
			ptr_pi[ll] += alpha*(ptr_dpi[ll] - ptr_pi[ll]);
		// box constraints
		for(ll=0; ll<nb0; ll++)
			{
			ptr_lam[ll+0] += alpha*ptr_dlam[ll+0];
			ptr_lam[ll+pnb] += alpha*ptr_dlam[ll+pnb];
			ptr_t[ll+0] += alpha*ptr_dt[ll+0];
			ptr_t[ll+pnb] += alpha*ptr_dt[ll+pnb];
			mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+pnb] * ptr_t[ll+pnb];
			}

		ptr_t    += 2*pnb;
		ptr_dt   += 2*pnb;
		ptr_lam  += 2*pnb;
		ptr_dlam += 2*pnb;

		// genreal constraints
		for(ll=0; ll<ng0; ll++)
			{
			ptr_lam[ll+0] += alpha*ptr_dlam[ll+0];
			ptr_lam[ll+png] += alpha*ptr_dlam[ll+png];
			ptr_t[ll+0] += alpha*ptr_dt[ll+0];
			ptr_t[ll+png] += alpha*ptr_dt[ll+png];
			mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+png] * ptr_t[ll+png];
			}

		}

	// scale mu
	mu *= mu_scal;

	ptr_mu[0] = mu;

	return;
	
	}



void d_update_var_soft_mpc_tv(int N, int *nx, int *nu, int *nb, int *ng, int *ns, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nu0, nx0, nb0, pnb, ng0, png, ns0, pns;

	int jj, ll;
	
	double
		*ptr_pi, *ptr_dpi, *ptr_ux, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lam, *ptr_dlam;

	double mu = 0;

	for(jj=0; jj<=N; jj++)
		{

		nx0 = nx[jj];
		nu0 = nu[jj];
		nb0 = nb[jj];
		pnb = bs*((nb0+bs-1)/bs); // cache aligned number of box constraints
		ng0 = ng[jj];
		png = bs*((ng0+bs-1)/bs); // cache aligned number of box constraints
		
		ptr_pi   = pi[jj];
		ptr_dpi  = dpi[jj];
		ptr_ux   = ux[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

		// update inputs and states
		for(ll=0; ll<nu0+nx0-3; ll+=4)
			{
			ptr_ux[ll+0] += alpha*(ptr_dux[ll+0] - ptr_ux[ll+0]);
			ptr_ux[ll+1] += alpha*(ptr_dux[ll+1] - ptr_ux[ll+1]);
			ptr_ux[ll+2] += alpha*(ptr_dux[ll+2] - ptr_ux[ll+2]);
			ptr_ux[ll+3] += alpha*(ptr_dux[ll+3] - ptr_ux[ll+3]);
			}
		for(; ll<nu0+nx0; ll++)

			ptr_ux[ll] += alpha*(ptr_dux[ll] - ptr_ux[ll]);
		// update equality constrained multipliers
		for(ll=0; ll<nx0-3; ll+=4)
			{
			ptr_pi[ll+0] += alpha*(ptr_dpi[ll+0] - ptr_pi[ll+0]);
			ptr_pi[ll+1] += alpha*(ptr_dpi[ll+1] - ptr_pi[ll+1]);
			ptr_pi[ll+2] += alpha*(ptr_dpi[ll+2] - ptr_pi[ll+2]);
			ptr_pi[ll+3] += alpha*(ptr_dpi[ll+3] - ptr_pi[ll+3]);
			}
		for(; ll<nx0; ll++)
			ptr_pi[ll] += alpha*(ptr_dpi[ll] - ptr_pi[ll]);

		// box constraints
		for(ll=0; ll<nb0; ll++)
			{
			ptr_lam[ll+0] += alpha*ptr_dlam[ll+0];
			ptr_lam[ll+pnb] += alpha*ptr_dlam[ll+pnb];
			ptr_t[ll+0] += alpha*ptr_dt[ll+0];
			ptr_t[ll+pnb] += alpha*ptr_dt[ll+pnb];
			mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+pnb] * ptr_t[ll+pnb];
			}

		ptr_t    += 2*pnb;
		ptr_dt   += 2*pnb;
		ptr_lam  += 2*pnb;
		ptr_dlam += 2*pnb;

		// genreal constraints
		for(ll=0; ll<ng0; ll++)
			{
			ptr_lam[ll+0] += alpha*ptr_dlam[ll+0];
			ptr_lam[ll+png] += alpha*ptr_dlam[ll+png];
			ptr_t[ll+0] += alpha*ptr_dt[ll+0];
			ptr_t[ll+png] += alpha*ptr_dt[ll+png];
			mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+png] * ptr_t[ll+png];
			}

		ptr_t    += 2*png;
		ptr_dt   += 2*png;
		ptr_lam  += 2*png;
		ptr_dlam += 2*png;

		// box soft constraints
		ns0 = ns[jj];
		pns  = bs*((ns0+bs-1)/bs); // cache aligned number of box soft constraints

		for(ll=0; ll<ns0; ll++)
			{
			ptr_lam[0*pns+ll] += alpha*ptr_dlam[0*pns+ll];
			ptr_lam[1*pns+ll] += alpha*ptr_dlam[1*pns+ll];
			ptr_lam[2*pns+ll] += alpha*ptr_dlam[2*pns+ll];
			ptr_lam[3*pns+ll] += alpha*ptr_dlam[3*pns+ll];
			ptr_t[0*pns+ll] += alpha*ptr_dt[0*pns+ll];
			ptr_t[1*pns+ll] += alpha*ptr_dt[1*pns+ll];
			ptr_t[2*pns+ll] += alpha*ptr_dt[2*pns+ll];
			ptr_t[3*pns+ll] += alpha*ptr_dt[3*pns+ll];
			mu += ptr_lam[0*pns+ll] * ptr_t[0*pns+ll] + ptr_lam[1*pns+ll] * ptr_t[1*pns+ll] + ptr_lam[2*pns+ll] * ptr_t[2*pns+ll] + ptr_lam[3*pns+ll] * ptr_t[3*pns+ll];
			}

		}

	// scale mu
	mu *= mu_scal;

	ptr_mu[0] = mu;

	return;
	
	}


void d_update_var_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{

	const int nbu = nu<nb ? nu : nb ;
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	//const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box and soft constraints
	const int pnb  = bs*((nb+bs-1)/bs); // cache aligned number of box and soft constraints
	const int png  = bs*((ng+bs-1)/bs); // simd aligned number of box and soft constraints
	const int pngN = bs*((ngN+bs-1)/bs); // simd aligned number of box and soft constraints at last stage

	int jj, ll;
	
	double
		*ptr_pi, *ptr_dpi, *ptr_ux, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lam, *ptr_dlam;

	double mu = 0;

	// first stage
	jj = 0;

	ptr_ux   = ux[jj];
	ptr_dux  = dux[jj];
	ptr_t    = t[jj];
	ptr_dt   = dt[jj];
	ptr_lam  = lam[jj];
	ptr_dlam = dlam[jj];

	// update inputs
	for(ll=0; ll<nu; ll++)
		ptr_ux[ll] += alpha*(ptr_dux[ll] - ptr_ux[ll]);
	// box constraints
	for(ll=0; ll<nbu; ll++)
		{
		ptr_lam[ll+0]   += alpha*ptr_dlam[ll+0];
		ptr_lam[ll+pnb] += alpha*ptr_dlam[ll+pnb];
		ptr_t[ll+0]   += alpha*ptr_dt[ll+0];
		ptr_t[ll+pnb] += alpha*ptr_dt[ll+pnb];
		mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+pnb] * ptr_t[ll+pnb];
		}

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

		// update inputs and states
		for(ll=0; ll<nu+nx; ll++)
			ptr_ux[ll] += alpha*(ptr_dux[ll] - ptr_ux[ll]);
		// update equality constrained multipliers
		for(ll=0; ll<nx; ll++)
			ptr_pi[ll] += alpha*(ptr_dpi[ll] - ptr_pi[ll]);
		// box constraints
		for(ll=0; ll<nb; ll++)
			{
			ptr_lam[ll+0] += alpha*ptr_dlam[ll+0];
			ptr_lam[ll+pnb] += alpha*ptr_dlam[ll+pnb];
			ptr_t[ll+0] += alpha*ptr_dt[ll+0];
			ptr_t[ll+pnb] += alpha*ptr_dt[ll+pnb];
			mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+pnb] * ptr_t[ll+pnb];
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

	// update states
	for(ll=0; ll<nx; ll++)
		ptr_ux[nu+ll] += alpha*(ptr_dux[nu+ll] - ptr_ux[nu+ll]);
	// update equality constrained multipliers
	for(ll=0; ll<nx; ll++)
		ptr_pi[ll] += alpha*(ptr_dpi[ll] - ptr_pi[ll]);
	// box constraints
	for(ll=nu; ll<nb; ll++)
		{
		ptr_lam[ll+0] += alpha*ptr_dlam[ll+0];
		ptr_lam[ll+pnb] += alpha*ptr_dlam[ll+pnb];
		ptr_t[ll+0] += alpha*ptr_dt[ll+0];
		ptr_t[ll+pnb] += alpha*ptr_dt[ll+pnb];
		mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+pnb] * ptr_t[ll+pnb];
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

			for(ll=2*pnb; ll<2*pnb+ng; ll++)
				{
				ptr_lam[ll+0] += alpha*ptr_dlam[ll+0];
				ptr_lam[ll+png] += alpha*ptr_dlam[ll+png];
				ptr_t[ll+0] += alpha*ptr_dt[ll+0];
				ptr_t[ll+png] += alpha*ptr_dt[ll+png];
				mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+png] * ptr_t[ll+png];
				}
			}

		}

	if(ngN>0)
		{

		ptr_t    = t[N];
		ptr_dt   = dt[N];
		ptr_lam  = lam[N];
		ptr_dlam = dlam[N];

		for(ll=2*pnb; ll<2*pnb+ngN; ll++)
			{
			ptr_lam[ll+0] += alpha*ptr_dlam[ll+0];
			ptr_lam[ll+pngN] += alpha*ptr_dlam[ll+pngN];
			ptr_t[ll+0] += alpha*ptr_dt[ll+0];
			ptr_t[ll+pngN] += alpha*ptr_dt[ll+pngN];
			mu += ptr_lam[ll+0] * ptr_t[ll+0] + ptr_lam[ll+pngN] * ptr_t[ll+pngN];
			}

		}

	// scale mu
	mu *= mu_scal;

	ptr_mu[0] = mu;

	return;
	
	}



void d_update_var_soft_mpc(int N, int nx, int nu, int nh, int ns, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{

	int nb = nh + ns;

	int nhu = nu<nh ? nu : nh ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	int jj, ll;
	
	double mu = 0;

	// initial stage
	// update inputs
	for(ll=0; ll<nu; ll++)
		ux[0][ll] += alpha*(dux[0][ll] - ux[0][ll]);
	// box constraints on inputs
	for(ll=0; ll<2*nhu; ll+=2)
		{
		lam[0][ll+0] += alpha*dlam[0][ll+0];
		lam[0][ll+1] += alpha*dlam[0][ll+1];
		t[0][ll+0] += alpha*dt[0][ll+0];
		t[0][ll+1] += alpha*dt[0][ll+1];
		mu += lam[0][ll+0] * t[0][ll+0] + lam[0][ll+1] * t[0][ll+1];
		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{
		// update inputs
		for(ll=0; ll<nu; ll++)
			ux[jj][ll] += alpha*(dux[jj][ll] - ux[jj][ll]);
		// update states
		for(ll=0; ll<nx; ll++)
			ux[jj][nu+ll] += alpha*(dux[jj][nu+ll] - ux[jj][nu+ll]);
		// update equality constrained multipliers
		for(ll=0; ll<nx; ll++)
			pi[jj][ll] += alpha*(dpi[jj][ll] - pi[jj][ll]);
		// box constraints on inputs and states
		ll = 0;
		for(; ll<2*nh; ll+=2)
			{
			lam[jj][ll+0] += alpha*dlam[jj][ll+0];
			lam[jj][ll+1] += alpha*dlam[jj][ll+1];
			t[jj][ll+0] += alpha*dt[jj][ll+0];
			t[jj][ll+1] += alpha*dt[jj][ll+1];
			mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1];
			}
		// soft constraints on states
		for(; ll<2*nb; ll+=2)
			{
			lam[jj][ll+0] += alpha*dlam[jj][ll+0];
			lam[jj][ll+1] += alpha*dlam[jj][ll+1];
			t[jj][ll+0] += alpha*dt[jj][ll+0];
			t[jj][ll+1] += alpha*dt[jj][ll+1];
			lam[jj][pnb+ll+0] += alpha*dlam[jj][pnb+ll+0];
			lam[jj][pnb+ll+1] += alpha*dlam[jj][pnb+ll+1];
			t[jj][pnb+ll+0] += alpha*dt[jj][pnb+ll+0];
			t[jj][pnb+ll+1] += alpha*dt[jj][pnb+ll+1];
			mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1] + lam[jj][pnb+ll+0] * t[jj][pnb+ll+0] + lam[jj][pnb+ll+1] * t[jj][pnb+ll+1];
			}
		}

	// final stage
	// update states
	for(ll=0; ll<nx; ll++)
		ux[N][nu+ll] += alpha*(dux[N][nu+ll] - ux[N][nu+ll]);
	// update equality constrained multipliers
	for(ll=0; ll<nx; ll++)
		pi[N][ll] += alpha*(dpi[N][ll] - pi[N][ll]);
	ll=2*nu;
	// box constraints on states
	for(; ll<2*nh; ll+=2)
		{
		lam[jj][ll+0] += alpha*dlam[jj][ll+0];
		lam[jj][ll+1] += alpha*dlam[jj][ll+1];
		t[jj][ll+0] += alpha*dt[jj][ll+0];
		t[jj][ll+1] += alpha*dt[jj][ll+1];
		mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1];
		}
	// soft constraints on states
	for(; ll<2*nb; ll+=2)
		{
		lam[N][ll+0] += alpha*dlam[N][ll+0];
		lam[N][ll+1] += alpha*dlam[N][ll+1];
		t[N][ll+0] += alpha*dt[N][ll+0];
		t[N][ll+1] += alpha*dt[N][ll+1];
		lam[N][pnb+ll+0] += alpha*dlam[N][pnb+ll+0];
		lam[N][pnb+ll+1] += alpha*dlam[N][pnb+ll+1];
		t[N][pnb+ll+0] += alpha*dt[N][pnb+ll+0];
		t[N][pnb+ll+1] += alpha*dt[N][pnb+ll+1];
		mu += lam[N][ll+0] * t[N][ll+0] + lam[N][ll+1] * t[N][ll+1] + lam[N][pnb+ll+0] * t[N][pnb+ll+0] + lam[N][pnb+ll+1] * t[N][pnb+ll+1];
		}
	mu *= mu_scal;

	ptr_mu[0] = mu;

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

		pnb  = bs*((nb[jj]+bs-1)/bs); // cache aligned number of box and soft constraints

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

	int nb0, pnb, ng0, png;

	int jj, ll;
	
	double
		*ptr_t, *ptr_lam, *ptr_dt, *ptr_dlam;
		
	double mu = 0;
	
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
		for(ll=0 ; ll<nb0; ll++)
			{
			mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+pnb] + alpha*ptr_dlam[ll+pnb]) * (ptr_t[ll+pnb] + alpha*ptr_dt[ll+pnb]);
			}

		ptr_t    += 2*pnb;
		ptr_dt   += 2*pnb;
		ptr_lam  += 2*pnb;
		ptr_dlam += 2*pnb;

		// general constraints
		for(ll=0; ll<ng0; ll++)
			{
			mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+png] + alpha*ptr_dlam[ll+png]) * (ptr_t[ll+png] + alpha*ptr_dt[ll+png]);
			}

		}

	// scale mu
	mu *= mu_scal;
		
	ptr_mu[0] = mu;

	return;

	}



void d_compute_mu_soft_mpc_tv(int N, int *nx, int *nu, int *nb, int *ng, int *ns, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nb0, pnb, ng0, png, ns0, pns;

	int jj, ll;
	
	double
		*ptr_t, *ptr_lam, *ptr_dt, *ptr_dlam;
		
	double mu = 0;
	
	for(jj=0; jj<=N; jj++)
		{
		
		nb0 = nb[jj];
		pnb = (nb0+bs-1)/bs*bs;
		
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

		// box constraints
		for(ll=0 ; ll<nb0; ll++)
			{
			mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+pnb] + alpha*ptr_dlam[ll+pnb]) * (ptr_t[ll+pnb] + alpha*ptr_dt[ll+pnb]);
			}

		ptr_t    += 2*pnb;
		ptr_dt   += 2*pnb;
		ptr_lam  += 2*pnb;
		ptr_dlam += 2*pnb;

		// general constraints
		ng0 = ng[jj];
		png = (ng0+bs-1)/bs*bs;
		for(ll=0; ll<ng0; ll++)
			{
			mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+png] + alpha*ptr_dlam[ll+png]) * (ptr_t[ll+png] + alpha*ptr_dt[ll+png]);
			}

		ptr_t    += 2*png;
		ptr_dt   += 2*png;
		ptr_lam  += 2*png;
		ptr_dlam += 2*png;

		// box soft constraints
		ns0 = ns[jj];
		pns  = bs*((ns0+bs-1)/bs); // cache aligned number of box soft constraints

		for(ll=0; ll<ns0; ll++)
			{
			mu += (ptr_lam[0*pns+ll] + alpha*ptr_dlam[0*pns+ll]) * (ptr_t[0*pns+ll] + alpha*ptr_dt[0*pns+ll]) + (ptr_lam[1*pns+ll] + alpha*ptr_dlam[1*pns+ll]) * (ptr_t[1*pns+ll] + alpha*ptr_dt[1*pns+ll]) + (ptr_lam[2*pns+ll] + alpha*ptr_dlam[2*pns+ll]) * (ptr_t[2*pns+ll] + alpha*ptr_dt[2*pns+ll]) + (ptr_lam[3*pns+ll] + alpha*ptr_dlam[3*pns+ll]) * (ptr_t[3*pns+ll] + alpha*ptr_dt[3*pns+ll]);
			}

		}

	// scale mu
	mu *= mu_scal;
		
	ptr_mu[0] = mu;

	return;

	}


void d_compute_mu_hard_mpc(int N, int nx, int nu, int nb, int ng, int ngN, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
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

	int jj, ll;
	
	double
		*ptr_t, *ptr_lam, *ptr_dt, *ptr_dlam;
		
	double mu = 0;
	
	// first stage
	jj = 0;

	ptr_t    = t[jj];
	ptr_lam  = lam[jj];
	ptr_dt   = dt[jj];
	ptr_dlam = dlam[jj];

	for(ll=0 ; ll<nbu; ll++)
		{
		mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+pnb] + alpha*ptr_dlam[ll+pnb]) * (ptr_t[ll+pnb] + alpha*ptr_dt[ll+pnb]);
		}


	for(jj=1; jj<N; jj++)
		{
		
		ptr_t    = t[jj];
		ptr_lam  = lam[jj];
		ptr_dt   = dt[jj];
		ptr_dlam = dlam[jj];

		for(ll=0 ; ll<nb; ll++)
			{
			mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+pnb] + alpha*ptr_dlam[ll+pnb]) * (ptr_t[ll+pnb] + alpha*ptr_dt[ll+pnb]);
			}
		}

	// last stage
	jj = N;
	
	ptr_t    = t[jj];
	ptr_lam  = lam[jj];
	ptr_dt   = dt[jj];
	ptr_dlam = dlam[jj];

	for(ll=nu ; ll<nb; ll++)
		{
		mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+pnb] + alpha*ptr_dlam[ll+pnb]) * (ptr_t[ll+pnb] + alpha*ptr_dt[ll+pnb]);
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

			for(ll=2*pnb; ll<2*pnb+ng; ll++)
				{
				mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+png] + alpha*ptr_dlam[ll+png]) * (ptr_t[ll+png] + alpha*ptr_dt[ll+png]);
				}
			}

		}
	if(ngN>0)
		{

		ptr_t    = t[N];
		ptr_lam  = lam[N];
		ptr_dt   = dt[N];
		ptr_dlam = dlam[N];

		for(ll=2*pnb; ll<2*pnb+ngN; ll++)
			{
			mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+pngN] + alpha*ptr_dlam[ll+pngN]) * (ptr_t[ll+pngN] + alpha*ptr_dt[ll+pngN]);
			}

		}

	// scale mu
	mu *= mu_scal;
		
	ptr_mu[0] = mu;

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
	
	double mu = 0;
	
	// fist stage: bounds on u only
	for(ll=0; ll<2*nhu; ll+=2)
		{
		mu += (lam[0][ll+0] + alpha*dlam[0][ll+0]) * (t[0][ll+0] + alpha*dt[0][ll+0]) + (lam[0][ll+1] + alpha*dlam[0][ll+1]) * (t[0][ll+1] + alpha*dt[0][ll+1]);
		}


	// middle stages: bounds on both u and x
	for(jj=1; jj<N; jj++)
		{
		for(ll=0; ll<2*nb; ll+=2)
			mu += (lam[jj][ll+0] + alpha*dlam[jj][ll+0]) * (t[jj][ll+0] + alpha*dt[jj][ll+0]) + (lam[jj][ll+1] + alpha*dlam[jj][ll+1]) * (t[jj][ll+1] + alpha*dt[jj][ll+1]);
		for(ll=pnb+2*nh; ll<pnb+2*nb; ll+=2)
			mu += (lam[jj][ll+0] + alpha*dlam[jj][ll+0]) * (t[jj][ll+0] + alpha*dt[jj][ll+0]) + (lam[jj][ll+1] + alpha*dlam[jj][ll+1]) * (t[jj][ll+1] + alpha*dt[jj][ll+1]);
		}	

	// last stage: bounds on x only
	for(ll=2*nu; ll<2*nb; ll+=2)
		mu += (lam[N][ll+0] + alpha*dlam[N][ll+0]) * (t[N][ll+0] + alpha*dt[N][ll+0]) + (lam[N][ll+1] + alpha*dlam[N][ll+1]) * (t[N][ll+1] + alpha*dt[N][ll+1]);
	for(ll=pnb+2*nh; ll<pnb+2*nb; ll+=2)
		mu += (lam[N][ll+0] + alpha*dlam[N][ll+0]) * (t[N][ll+0] + alpha*dt[N][ll+0]) + (lam[N][ll+1] + alpha*dlam[N][ll+1]) * (t[N][ll+1] + alpha*dt[N][ll+1]);

	mu *= mu_scal;
		
	ptr_mu[0] = mu;

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




