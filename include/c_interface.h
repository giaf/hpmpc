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

#include "target.h"
#include "block_size.h"



// MPC problem: define NX (state dimension), NU (input dimension), NN (horizon length)
// MHE problem: define NX (state dimension), NU (process disturbance dimension), NY (measurement dimension), NN (horizon length)


// define common quantities
#define NZ (NX+NU+1)
#define NT (NX+NY)
#define NB (NU+NX) // number of two-sided box constraints
// double precision constants
#define D_NAL (D_MR*D_NCL)
#define D_PADX ((D_NCL-NX%D_NCL)%D_NCL) // padding after nx
#define D_PADXU ((D_NCL-(NX+NU)%D_NCL)%D_NCL) // padding after (nx+nw)
#define D_ANB (D_NAL*((2*NB+D_NAL-1)/(D_NAL)))
#define D_ANT (D_NAL*((NT+D_NAL-1)/(D_NAL)))
#define D_ANU (D_NAL*((NU+D_NAL-1)/(D_NAL)))
#define D_ANX (D_NAL*((NX+D_NAL-1)/(D_NAL)))
#define D_ANY (D_NAL*((NY+D_NAL-1)/(D_NAL)))
#define D_ANZ (D_NAL*((NZ+D_NAL-1)/(D_NAL)))
#define D_CNF (D_CNT<D_CNX+D_NCL ? D_CNX+D_NCL : D_CNT)
#define D_CNJ (NX+NU+D_PADXU+D_CNX)
#define D_CNL (D_CNZ<D_CNX+D_NCL ? NX+D_PADX+D_CNX+D_NCL : NX+D_PADX+D_CNZ)
#define D_CNT (D_NCL*((NT+D_NCL-1)/D_NCL))
#define D_CNU (D_NCL*((NU+D_NCL-1)/D_NCL))
#define D_CNX (D_NCL*((NX+D_NCL-1)/D_NCL))
#define D_CNY (D_NCL*((NY+D_NCL-1)/D_NCL))
#define D_CNZ (D_NCL*((NZ+D_NCL-1)/D_NCL))
#define D_CNX2 (2*D_NCL*((NX+D_NCL-1)/D_NCL))
#define D_PNB (D_MR*((2*NB+D_MR-1)/D_MR))
#define D_PNT (D_MR*((NT+D_MR-1)/D_MR))
#define D_PNU (D_MR*((NU+D_MR-1)/D_MR))
#define D_PNX (D_MR*((NX+D_MR-1)/D_MR))
#define D_PNY (D_MR*((NY+D_MR-1)/D_MR))
#define D_PNZ (D_MR*((NZ+D_MR-1)/D_MR))
#define D_PNX2 (D_MR*((NX+NX+D_MR-1)/D_MR))
#define D_PNUX (D_MR*((NU+NX+D_MR-1)/D_MR))
// single precision constants
#define S_NAL (S_MR*S_NCL)
#define S_PADX ((S_NCL-NX%S_NCL)%S_NCL) // padding between BAbtL & P
#define S_ANB (S_NAL*((2*NB+S_NAL-1)/(S_NAL)))
#define S_ANX (S_NAL*((NX+S_NAL-1)/(S_NAL)))
#define S_ANZ (S_NAL*((NZ+S_NAL-1)/(S_NAL)))
#define S_CNL (S_CNZ<S_CNX+S_NCL ? NX+S_PADX+S_CNX+S_NCL : NX+S_PADX+S_CNZ)
#define S_CNX (S_NCL*((NX+S_NCL-1)/S_NCL))
#define S_CNZ (S_NCL*((NZ+S_NCL-1)/S_NCL))
#define S_PNB (S_MR*((2*NB+S_MR-1)/S_MR))
#define S_PNX (S_MR*((NX+S_MR-1)/S_MR))
#define S_PNZ (S_MR*((NZ+S_MR-1)/S_MR))



// work space: static definition

// Riccati-based IP method for box-constrained MPC, double precision
#define HPMPC_IP_MPC_DP_WORK_SPACE (8 + (NN+1)*(D_PNZ*D_CNX + D_PNZ*D_CNZ + D_PNZ*D_CNL + 6*D_ANZ + 3*D_ANX + 7*D_ANB) + 3*D_ANZ)
// Riccati-based IP method for box-constrained MPC, single precision
#define HPMPC_IP_MPC_SP_WORK_SPACE (16 + (NN+1)*(S_PNZ*S_CNX + S_PNZ*S_CNZ + S_PNZ*S_CNL + 5*S_ANZ + 3*S_ANX + 7*S_ANB) + 3*S_ANZ)
// Riccati-based IP method for soft-constrained MPC, double precision
#define HPMPC_IP_SOFT_MPC_DP_WORK_SPACE (8 + (NN+1)*(D_PNZ*D_CNX + D_PNZ*D_CNZ + D_PNZ*D_CNL + 6*D_ANZ + 3*D_ANX + 17*D_ANB) + 3*D_ANZ)
// Riccati-based solver for unconstrained MPC, double precision
#define HPMPC_RIC_MPC_DP_WORK_SPACE (8 + (NN+1)*(D_PNZ*D_CNX + D_PNZ*D_CNZ + D_PNZ*D_CNL + 2*D_ANZ + 2*D_ANX) + 3*D_ANZ)
// Riccati-based solver for unconstrained MPC, single precision
#define HPMPC_RIC_MPC_SP_WORK_SPACE (16 + (NN+1)*(S_PNZ*S_CNX + S_PNZ*S_CNZ + S_PNZ*S_CNL + 2*S_ANZ + 2*S_ANX) + 3*S_ANZ)
// Riccati-based solver for unconstrained MHE, double precision
#define HPMPC_RIC_MHE_DP_WORK_SPACE (8 + (NN+1)*(D_PNX*D_CNX+D_PNX*D_CNU+D_PNY*D_CNX+5*D_ANX+D_PNU*D_CNU+D_PNY*D_CNY+2*D_ANU+2*D_ANY+D_PNX*D_CNJ+D_PNT*D_CNF) + 2*D_PNY*D_CNX+D_PNT*D_CNT+D_ANT+D_PNU*D_CNU+D_PNX*D_CNX)
// Riccati-based solver for unconstrained MHE, Information Filter version, double precision
#define HPMPC_RIC_MHE_IF_DP_WORK_SPACE (8 + (NN+1)*(D_PNUX*D_CNU+D_PNX2*D_CNX+D_PNUX*D_CNU+D_PNX2*D_CNX2+D_PNX*D_CNY+2*D_ANU+D_ANY+5*D_ANX) + 2*D_PNX*D_CNX+D_PNX*D_CNJ+D_ANX+D_PNY*D_CNY+D_PNX*D_CNY+D_ANX)

// work space: dynamic definition as function return value

// Riccati-based IP method for box-constrained MPC, double precision
int hpmpc_ip_box_mpc_dp_work_space(int nx, int nu, int N);
#if 0
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
	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int anb = nal*((2*nb+nal-1)/nal);

	int work_space_size = (8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 6*anz + 3*anx + 7*anb) + 3*anz);

	return work_space_size;
	}
#endif

// Riccati-based IP method for box-constrained MPC, single precision
int hpmpc_ip_box_mpc_sp_work_space(int nx, int nu, int N);
#if 0
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
	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int anb = nal*((2*nb+nal-1)/nal);

	int work_space_size = (16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz);

	return work_space_size;
	}
#endif
    
// Riccati-based IP method for box-constrained MPC, double precision
int hpmpc_ip_soft_mpc_dp_work_space(int nx, int nu, int N);

// Riccati-based solver for unconstrained MPC, double precision
int hpmpc_ric_mpc_dp_work_space(int nx, int nu, int N);
#if 0
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
	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

	int work_space_size = (8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + 3*anz);

	return work_space_size;
	}
#endif

// Riccati-based solver for unconstrained MPC, single precision
int hpmpc_ric_mpc_sp_work_space(int nx, int nu, int N);
#if 0
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
	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

	int work_space_size = (16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + 3*anz);

	return work_space_size;
	}
#endif

// Riccati-based solver for unconstrained MHE, covariance filter version, double precision
int hpmpc_ric_mhe_dp_work_space(int nx, int nw, int ny, int N);
#if 0
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
#endif

// Riccati-based solver for unconstrained MHE, information filter version, double precision
int hpmpc_ric_mhe_if_dp_work_space(int nx, int nw, int ny, int N);
#if 0
	{
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = D_MR*D_NCL;
	const int nwx = nw+nx; 
	const int anx = nal*((nx+nal-1)/nal);
	const int anw = nal*((nw+nal-1)/nal);
	const int any = nal*((ny+nal-1)/nal);
	const int pnx = bs*((nx+bs-1)/bs);
	const int pny = bs*((ny+bs-1)/bs);
	const int pnx2 = bs*((2*nx+bs-1)/bs);
	const int pnwx = bs*((nwx+bs-1)/bs);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);
	const int cny = ncl*((ny+ncl-1)/ncl);
	const int cnx2 = 2*(ncl*((nx+ncl-1)/ncl));
	const int pad = (ncl-(nx+nw)%ncl)%ncl; // padding
	const int cnj = nx+nw+pad+cnx;

	int work_space_size = (8 + (N+1)*(pnwx*cnw+pnx2*cnx+pnwx*cnw+pnx2*cnx2+pnx*cny+2*anw+any+5*anx) + 2*pnx*cnx+pnx*cnj+anx+pny*cny+pnx*cny+anx);

	return work_space_size;
	}
#endif



// c (or row-major) order
int c_order_ip_box_mpc( int k_max, double mu_tol, const char prec, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lb, double* ub, double* x, double* u, double *work0, int* nIt, double *stat );

int c_order_ip_soft_mpc( int k_max, double mu_tol, const char prec, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lZ, double* uZ, double* lz, double* uz, double* lb, double* ub, double* x, double* u, double *work0, int* nIt, double *stat );

int c_order_riccati_mpc( const char prec, const int nx, const int nu, const int N, double *A, double *B, double *b, double *Q, double *Qf, double *S, double *R, double *q, double *qf, double *r, double *x, double *u, double *pi, double *work0 );

int c_order_riccati_mhe( const char prec, const int smooth, const int nx, const int nw, const int ny, const int N, double *A, double *G, double *C, double *f, double *Q, double *R, double *q, double *r, double *y, double *x0, double *L0, double *xe, double *Le, double *w, double *lam, double *work0 );

int c_order_riccati_mhe_if( char prec, int alg, int nx, int nw, int ny, int N, double *A, double *G, double *C, double *f, double *R, double *Q, double *Qf, double *r, double *q, double *qf, double *y, double *x0, double *L0, double *xe, double *Le, double *w, double *lam, double *work0 );



// fortran (or column-major) order
int fortran_order_ip_box_mpc( int k_max, double mu_tol, const char prec, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lb, double* ub, double* x, double* u, double *work0, int* nIt, double *stat );

int fortran_order_ip_soft_mpc( int k_max, double mu_tol, const char prec, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lZ, double* uZ, double* lz, double* uz, double* lb, double* ub, double* x, double* u, double *work0, int* nIt, double *stat );

int fortran_order_riccati_mpc( const char prec, const int nx, const int nu, const int N, double *A, double *B, double *b, double *Q, double *Qf, double *S, double *R, double *q, double *qf, double *r, double *x, double *u, double *pi, double *work0 );

int fortran_order_riccati_mhe( const char prec, const int smooth, const int nx, const int nw, const int ny, const int N, double *A, double *G, double *C, double *f, double *Q, double *R, double *q, double *r, double *y, double *x0, double *L0, double *xe, double *Le, double *w, double *lam, double *work0 );

int fortran_order_riccati_mhe_if( char prec, int alg, int nx, int nw, int ny, int N, double *A, double *G, double *C, double *f, double *R, double *Q, double *Qf, double *r, double *q, double *qf, double *y, double *x0, double *L0, double *xe, double *Le, double *w, double *lam, double *work0 );
