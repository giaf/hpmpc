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
int hpmpc_ip_hard_mpc_dp_work_space(int N, int nx, int nu, int nb, int ng, int ngN)
	{
	//const int nb = nx+nu;
	//const int ng = 0;

	const int bs  = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = D_MR*D_NCL;

	const int nz   = nx+nu+1;
	const int pnz  = bs*((nz+bs-1)/bs);
	const int pnb  = bs*((nb+bs-1)/bs);
	const int png  = bs*((ng+bs-1)/bs);
	const int pngN = bs*((ngN+bs-1)/bs);
	const int cnz  = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx  = ncl*((nx+ncl-1)/ncl);
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);
	const int cnxg = ncl*((nx+ng+ncl-1)/ncl);
	const int anz  = nal*((nz+nal-1)/nal);
	const int anx  = nal*((nx+nal-1)/nal);
//	const int pad  = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
//	const int cnl  = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int cnl  = cnz<cnx+ncl ? cnx+ncl : cnz;
	//const int anb = 2*nal*((nb+nal-1)/nal);

//	int work_space_size = (8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 6*anz + 3*anx + 7*anb) + 3*anz);
	int work_space_size = (8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + pnz*cng + 8*anz + 4*anx + 18*(pnb+png)) + pnz*(cngN-cng) + 18*(pngN-png) + anz + (cngN<cnxg ? pnz*cnxg : pnz*cngN) );

	return work_space_size;
	}

// Riccati-based IP method for box-constrained MPC, single precision
int hpmpc_ip_box_mpc_sp_work_space(int nx, int nu, int N)
	{
	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;
	const int nal = S_MR*S_NCL;
	const int nz = nx+nu+1;
	const int nb = nx+nu; // number of two-sided box constraints
	const int pnz = bs*((nz+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int anz = nal*((nz+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);
//	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
//	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;
	const int anb = 2*nal*((nb+nal-1)/nal);

//	int work_space_size = (16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz);
	int work_space_size = (16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + anz + pnz*cnx);

	return work_space_size;
	}
    
// Riccati-based IP method for soft-constrained MPC, double precision
int hpmpc_ip_soft_mpc_dp_work_space(int nx, int nu, int N)
	{
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = D_MR*D_NCL;
	const int nz = nx+nu+1;
	const int nb = nx+nu; // number of two-sided box constraints
	const int pnz = bs*((nz+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int anz = nal*((nz+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);
//	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
//	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;
	const int anb = 2*nal*((nb+nal-1)/nal);

//	int work_space_size = (8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 6*anz + 3*anx + 17*anb) + 3*anz);
	int work_space_size = (8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 6*anz + 3*anx + 17*anb) + anz + pnz*cnx);

	return work_space_size;
	}

// Riccati-based solver for unconstrained MPC, double precision
int hpmpc_ric_mpc_dp_work_space(int nx, int nu, int N)
	{
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = D_MR*D_NCL;
	const int nz = nx+nu+1;
	const int pnz = bs*((nz+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int anz = nal*((nz+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);
//	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
//	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;

//	int work_space_size = (8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + 3*anz);
	int work_space_size = (8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + anz + pnz*cnx);

	return work_space_size;
	}

// Riccati-based solver for unconstrained MPC, single precision
int hpmpc_ric_mpc_sp_work_space(int nx, int nu, int N)
	{
	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;
	const int nal = S_MR*S_NCL;
	const int nz = nx+nu+1;
	const int pnz = bs*((nz+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int anz = nal*((nz+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);
//	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
//	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;

//	int work_space_size = (16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + 3*anz);
	int work_space_size = (16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + anz + pnz*cnx);

	return work_space_size;
	}

// Riccati-based solver for unconstrained MHE, covariance filter version, double precision
int hpmpc_ric_mhe_dp_work_space(int nx, int nw, int ny, int N)
	{
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = D_MR*D_NCL;
	const int nt = nx+ny; 
	const int ant = nal*((nt+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);
	const int anw = nal*((nw+nal-1)/nal);
	const int any = nal*((ny+nal-1)/nal);
	const int pnt = bs*((nt+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int pnw = bs*((nw+bs-1)/bs);
	const int pny = bs*((ny+bs-1)/bs);
	const int cnt = ncl*((nt+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);
	const int cny = ncl*((ny+ncl-1)/ncl);
	const int cnf = cnt<cnx+ncl ? cnx+ncl : cnt;
	const int pad = (ncl-(nx+nw)%ncl)%ncl; // packing between AGL & P
	const int cnj = nx+nw+pad+cnx;

	int work_space_size = (8 + (N+1)*(pnx*cnx+pnx*cnw+pny*cnx+5*anx+pnw*cnw+pny*cny+2*anw+2*any+pnx*cnj+pnt*cnf) + 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx);

	return work_space_size;
	}

// Riccati-based solver for unconstrained MHE, information filter version, double precision
int hpmpc_ric_mhe_if_dp_work_space(int nx, int nw, int ny, int ndN, int N)
	{
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = D_MR*D_NCL;
	const int nwx = nw+nx; 
	const int anx = nal*((nx+nal-1)/nal);
	const int anw = nal*((nw+nal-1)/nal);
	const int any = nal*((ny+nal-1)/nal);
	const int pnx = bs*((nx+bs-1)/bs);
	const int pnw = bs*((nw+bs-1)/bs);
	const int pny = bs*((ny+bs-1)/bs);
	const int pnx2 = bs*((2*nx+bs-1)/bs);
	const int pnwx = bs*((nwx+bs-1)/bs);
	const int pndN = bs*((ndN+bs-1)/bs);
//	const int pnxdN = bs*((nx+ndN+bs-1)/bs);
	const int pnwx1 = pnx>pnw ? 2*pnx : pnx+pnw;
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);
	const int cny = ncl*((ny+ncl-1)/ncl);
	const int cnx2 = 2*(ncl*((nx+ncl-1)/ncl));
	const int pad = (ncl-(nx+nw)%ncl)%ncl; // padding
	const int cnj = nx+nw+pad+cnx;
	const int cndN = ncl*((ndN+ncl-1)/ncl);
	const int cnwx1 = ncl*((nw+nx+1+ncl-1)/ncl);
	const int pnm = pnx>pnw ? pnx : pnw;

	//int work_space_size = (8 + (N+1)*(pnwx*cnw+pnx2*cnx+pnwx*cnw+pnx2*cnx2+pnx*cny+2*anw+any+5*anx) + 2*pnx*cnx+pnx*cnj+anx+pny*cny+pnx*cny+anx + pndN*cndN);

	int work_space_size = (8 + (N+1)*(2*pnwx1*cnwx1+pnx*cny+pndN*cndN+pnx*cnx+5*anx+2*anw+any) + pny*cny+pnx*cny+2*anx+pnx*cnx+pnm+any);

	return work_space_size;
	}


