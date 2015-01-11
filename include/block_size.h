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

#ifndef __HPMPC_BLOCK_SIZE__
#define __HPMPC_BLOCK_SIZE__

#if defined( TARGET_X64_AVX2 )

#define D_MR 4
#define S_MR 8
#define D_NCL 2
#define S_NCL 2

#elif defined( TARGET_X64_AVX )

#define D_MR 4
#define S_MR 8
#define D_NCL 2
#define S_NCL 2

#elif defined( TARGET_X64_SSE3 )

#define D_MR 4
#define S_MR 4
#define D_NCL 2
#define S_NCL 4

#elif defined( TARGET_C99_4X4 )

#define D_MR 4
#define S_MR 4
#define D_NCL 2
#define S_NCL 4

#elif defined( TARGET_C99_4X4_PREFETCH )

#define D_MR 4
#define S_MR 4
#define D_NCL 2
#define S_NCL 4

#elif defined( TARGET_CORTEX_A15 )

#define D_MR 4
#define S_MR 4
#define D_NCL 2
#define S_NCL 4

#elif defined( TARGET_CORTEX_A9 )

#define D_MR 4
#define S_MR 4
#define D_NCL 2 // 1
#define S_NCL 2

#elif defined( TARGET_CORTEX_A7 )

#define D_MR 4
#define S_MR 4
#define D_NCL 2
#define S_NCL 4


#else
#error "Unknown architecture"
#endif /* __HPMPC_BLOCK_SIZE__ */

int d_get_mr();
int s_get_mr();

#endif /* __HPMPC_BLOCK_SIZE__ */
