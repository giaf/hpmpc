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



#if 0
void d_update_diag_pmat(int kmax, double *pQ, int sda, double *d)
	{

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pQ[jj*sda+(jj+0)*4+0] = d[jj+0];
		pQ[jj*sda+(jj+1)*4+1] = d[jj+1];
		pQ[jj*sda+(jj+2)*4+2] = d[jj+2];
		pQ[jj*sda+(jj+3)*4+3] = d[jj+3];
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pQ[jj*sda+(jj+ll)*4+ll] = d[jj+ll];
		}
	
	}



void d_update_diag_pmat_sparse(int kmax, int *idx, double *pQ, int sda, double *d)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pQ[ii/bs*bs*sda+ii%bs+ii*bs] = d[jj];
		}
	
	}



void d_update_row_pmat(int kmax, double *pQ, double *r)
	{

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pQ[(jj+0)*4] = r[jj+0];
		pQ[(jj+1)*4] = r[jj+1];
		pQ[(jj+2)*4] = r[jj+2];
		pQ[(jj+3)*4] = r[jj+3];
		}
	for(; jj<kmax; jj++)
		{
		pQ[(jj)*4] = r[jj];
		}
	
	}



void d_update_row_pmat_sparse(int kmax, int *idx, double *pQ, double *r)
	{

	const int bs = 4;

	int ii, jj;

	for(jj=0; jj<kmax; jj++)
		{
		ii = idx[jj];
		pQ[ii*bs] = r[jj];
		}
	
	}



void d_add_diag_pmat(int kmax, double *pQ, int sda, double *d)
	{

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pQ[jj*sda+(jj+0)*4+0] += d[jj+0];
		pQ[jj*sda+(jj+1)*4+1] += d[jj+1];
		pQ[jj*sda+(jj+2)*4+2] += d[jj+2];
		pQ[jj*sda+(jj+3)*4+3] += d[jj+3];
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pQ[jj*sda+(jj+ll)*4+ll] += d[jj+ll];
		}
	
	}



void d_add_row_pmat(int kmax, double *pA, double *pC)
	{

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pC[(jj+0)*4] += pA[(jj+0)*4];
		pC[(jj+1)*4] += pA[(jj+1)*4];
		pC[(jj+2)*4] += pA[(jj+2)*4];
		pC[(jj+3)*4] += pA[(jj+3)*4];
		}
	for(; jj<kmax; jj++)
		{
		pC[(jj+0)*4] += pA[(jj+0)*4];
		}
	
	}
#endif



void d_update_vector_sparse(int kmax, int *idx, double *q, double *v)
	{

	int jj;

	for(jj=0; jj<kmax; jj++)
		{
		q[idx[jj]] = v[jj];
		}
	
	}



// the diagonal is inverted !!!!!
void dpotrf_diag_lib(int m, int n, double *C, int sdc, double *D, int sdd)
	{

	const int bs = 4;

	int ii, jj, ll, ll_max;

	double alpha0, alpha1, alpha2, alpha3;

	for(jj=0; jj<n-3; jj+=4)
		{
		alpha0 = 1.0/sqrt(C[(jj)*sdc+0+(jj+0)*bs]);
		alpha1 = 1.0/sqrt(C[(jj)*sdc+1+(jj+1)*bs]);
		alpha2 = 1.0/sqrt(C[(jj)*sdc+2+(jj+2)*bs]);
		alpha3 = 1.0/sqrt(C[(jj)*sdc+3+(jj+3)*bs]);
		// initial triangle
		D[(jj)*sdc+0+(jj+0)*bs] = alpha0;
		D[(jj)*sdc+1+(jj+0)*bs] = 0.0;
		D[(jj)*sdc+2+(jj+0)*bs] = 0.0;
		D[(jj)*sdc+3+(jj+0)*bs] = 0.0;
		D[(jj)*sdc+1+(jj+1)*bs] = alpha1;
		D[(jj)*sdc+2+(jj+1)*bs] = 0.0;
		D[(jj)*sdc+3+(jj+1)*bs] = 0.0;
		D[(jj)*sdc+2+(jj+2)*bs] = alpha2;
		D[(jj)*sdc+3+(jj+2)*bs] = 0.0;
		D[(jj)*sdc+3+(jj+3)*bs] = alpha3;
		// remaining rows
		for(ll=jj+4; ll<n/bs*bs; ll+=4)
			{
			D[(ll)*sdd+0+(jj+0)*bs] = 0.0;
			D[(ll)*sdd+1+(jj+0)*bs] = 0.0;
			D[(ll)*sdd+2+(jj+0)*bs] = 0.0;
			D[(ll)*sdd+3+(jj+0)*bs] = 0.0;

			D[(ll)*sdd+0+(jj+1)*bs] = 0.0;
			D[(ll)*sdd+1+(jj+1)*bs] = 0.0;
			D[(ll)*sdd+2+(jj+1)*bs] = 0.0;
			D[(ll)*sdd+3+(jj+1)*bs] = 0.0;

			D[(ll)*sdd+0+(jj+2)*bs] = 0.0;
			D[(ll)*sdd+1+(jj+2)*bs] = 0.0;
			D[(ll)*sdd+2+(jj+2)*bs] = 0.0;
			D[(ll)*sdd+3+(jj+2)*bs] = 0.0;

			D[(ll)*sdd+0+(jj+3)*bs] = 0.0;
			D[(ll)*sdd+1+(jj+3)*bs] = 0.0;
			D[(ll)*sdd+2+(jj+3)*bs] = 0.0;
			D[(ll)*sdd+3+(jj+3)*bs] = 0.0;
			}
		for(; ll<m; ll+=4)
			{
			D[(ll)*sdd+0+(jj+0)*bs] = alpha0 * C[(ll)*sdc+0+(jj+0)*bs];
			D[(ll)*sdd+1+(jj+0)*bs] = alpha0 * C[(ll)*sdc+1+(jj+0)*bs];
			D[(ll)*sdd+2+(jj+0)*bs] = alpha0 * C[(ll)*sdc+2+(jj+0)*bs];
			D[(ll)*sdd+3+(jj+0)*bs] = alpha0 * C[(ll)*sdc+3+(jj+0)*bs];

			D[(ll)*sdd+0+(jj+1)*bs] = alpha1 * C[(ll)*sdc+0+(jj+1)*bs];
			D[(ll)*sdd+1+(jj+1)*bs] = alpha1 * C[(ll)*sdc+1+(jj+1)*bs];
			D[(ll)*sdd+2+(jj+1)*bs] = alpha1 * C[(ll)*sdc+2+(jj+1)*bs];
			D[(ll)*sdd+3+(jj+1)*bs] = alpha1 * C[(ll)*sdc+3+(jj+1)*bs];

			D[(ll)*sdd+0+(jj+2)*bs] = alpha2 * C[(ll)*sdc+0+(jj+2)*bs];
			D[(ll)*sdd+1+(jj+2)*bs] = alpha2 * C[(ll)*sdc+1+(jj+2)*bs];
			D[(ll)*sdd+2+(jj+2)*bs] = alpha2 * C[(ll)*sdc+2+(jj+2)*bs];
			D[(ll)*sdd+3+(jj+2)*bs] = alpha2 * C[(ll)*sdc+3+(jj+2)*bs];

			D[(ll)*sdd+0+(jj+3)*bs] = alpha3 * C[(ll)*sdc+0+(jj+3)*bs];
			D[(ll)*sdd+1+(jj+3)*bs] = alpha3 * C[(ll)*sdc+1+(jj+3)*bs];
			D[(ll)*sdd+2+(jj+3)*bs] = alpha3 * C[(ll)*sdc+2+(jj+3)*bs];
			D[(ll)*sdd+3+(jj+3)*bs] = alpha3 * C[(ll)*sdc+3+(jj+3)*bs];
			}
		}
	for(ii=0; ii<n-jj; ii++)
		{
		alpha0 = 1.0/sqrt(C[(jj)*sdc+ii+(jj+ii)*bs]);
		D[(jj)*sdc+ii+(jj+ii)*bs] = alpha0;
		// initial part of column
		ll_max = n-jj<4 ? n-jj : 4;
		for(ll=ii+1; ll<ll_max; ll++)
			D[(jj)*sdc+ll+(jj+ii)*bs] = 0.0;
		for(; ll<4; ll++)
			D[(jj)*sdc+ll+(jj+ii)*bs] = alpha0 * C[(jj)*sdc+ll+(jj+ii)*bs];
		// remaining
		for(ll=jj+4; ll<n/bs*bs+8; ll+=4)
			{
			D[(ll)*sdd+0+(jj+ii)*bs] = 0.0;
			D[(ll)*sdd+1+(jj+ii)*bs] = 0.0;
			D[(ll)*sdd+2+(jj+ii)*bs] = 0.0;
			D[(ll)*sdd+3+(jj+ii)*bs] = 0.0;
			}
		for(; ll<m; ll+=4)
			{
			D[(ll)*sdd+0+(jj+ii)*bs] = alpha0 * C[(ll)*sdc+0+(jj+ii)*bs];
			D[(ll)*sdd+1+(jj+ii)*bs] = alpha0 * C[(ll)*sdc+1+(jj+ii)*bs];
			D[(ll)*sdd+2+(jj+ii)*bs] = alpha0 * C[(ll)*sdc+2+(jj+ii)*bs];
			D[(ll)*sdd+3+(jj+ii)*bs] = alpha0 * C[(ll)*sdc+3+(jj+ii)*bs];
			}
		}

	}

