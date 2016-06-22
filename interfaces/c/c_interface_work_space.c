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

#include "../../include/target.h"
#include "../../include/block_size.h"



// work space: dynamic definition as function return value

// Riccati-based IP method for box-constrained MPC, double precision
// XXX assume size(double) >= size(int) && size(double) >= size(int *) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
int hpmpc_d_ip_mpc_hard_tv_work_space_size_bytes(int N, int nx, int nu, int nb, int ng, int ngN)
	{

	const int bs  = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	const int nz   = nx+nu+1;
	const int pnz  = bs*((nz+bs-1)/bs);
	const int pnx  = (nx+bs-1)/bs*bs;
	const int pnb  = bs*((nb+bs-1)/bs);
	const int png  = bs*((ng+bs-1)/bs);
	const int pngN = bs*((ngN+bs-1)/bs);
	const int cnz  = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx  = ncl*((nx+ncl-1)/ncl);
	const int cnux = (nu+nx+ncl-1)/ncl*ncl;
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);
	const int cnxg = ncl*((nx+ng+ncl-1)/ncl);
	const int cnl  = cnux<cnx+ncl ? cnx+ncl : cnux;

	int work_space_size = (8 + bs + (N+1)*(nb + pnz*cnx + pnz*cnux + pnz*cnl + pnz*cng + 7*pnz + 6*pnx + 23*pnb + 19*png) + pnz*(cngN-cng) + 19*(pngN-png) + pnz + (cngN<cnxg ? pnz*cnxg : pnz*cngN) );

	return work_space_size*sizeof(double);

	}



int hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(int N, int const *nx, int const *nu, int const *nb, int const *ng)
	{

	const int bs  = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	int ii;

	int pnx[N+1];
	int pnz[N+1];
	int pnb[N+1];
	int png[N+1];
	int cnx[N+1];
	int cnux[N+1];
	int cng[N+1];

	for(ii=0; ii<=N; ii++)
		{
		pnx[ii] = (nx[ii]+bs-1)/bs*bs;
		pnz[ii] = (nu[ii]+nx[ii]+1+bs-1)/bs*bs;
		pnb[ii] = (nb[ii]+bs-1)/bs*bs;
		png[ii] = (ng[ii]+bs-1)/bs*bs;
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		cnux[ii] = (nu[ii]+nx[ii]+ncl-1)/ncl*ncl;
		cng[ii] = (ng[ii]+ncl-1)/ncl*ncl;
		}

	int size_doubles = bs;
	int size_ints = 0;

	for(ii=0; ii<N; ii++)
		{
		size_doubles += pnz[ii]*cnx[ii+1] + pnz[ii]*cng[ii] + pnz[ii]*cnux[ii] + 3*pnx[ii] + 3*pnz[ii] + 8*pnb[ii] + 8*png[ii];
		size_ints += nb[ii];
		}
	ii = N;
	size_doubles += pnz[ii]*cng[ii] + pnz[ii]*cnux[ii] + 3*pnx[ii] + 3*pnz[ii] + 8*pnb[ii] + 8*png[ii];
	size_ints += nb[ii];

	int work_space_size = 64 + d_ip2_res_mpc_hard_tv_work_space_size_bytes(N, nx, nu, nb, ng) + size_doubles*sizeof(double) + size_ints*sizeof(int);

	return work_space_size;

	}


