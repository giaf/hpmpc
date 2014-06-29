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

#include <stdio.h>



int main()
	{
	
	FILE *f;
    f = fopen("./include/target.h", "w"); // a

	fprintf(f, "/**************************************************************************************************\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* This file is part of HPMPC.                                                                     *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *\n");
	fprintf(f, "* Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* HPMPC is free software; you can redistribute it and/or                                          *\n");
	fprintf(f, "* modify it under the terms of the GNU Lesser General Public                                      *\n");
	fprintf(f, "* License as published by the Free Software Foundation; either                                    *\n");
	fprintf(f, "* version 2.1 of the License, or (at your option) any later version.                              *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* HPMPC is distributed in the hope that it will be useful,                                        *\n");
	fprintf(f, "* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *\n");
	fprintf(f, "* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *\n");
	fprintf(f, "* See the GNU Lesser General Public License for more details.                                     *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* You should have received a copy of the GNU Lesser General Public                                *\n");
	fprintf(f, "* License along with HPMPC; if not, write to the Free Software                                    *\n");
	fprintf(f, "* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "**************************************************************************************************/\n");
	fprintf(f, "\n");
#if defined(TARGET_X64_AVX2)
	fprintf(f, "#ifndef TARGET_X64_AVX2\n");
	fprintf(f, "#define TARGET_X64_AVX2\n");
	fprintf(f, "#endif\n");
#elif defined(TARGET_X64_AVX)
	fprintf(f, "#ifndef TARGET_X64_AVX\n");
	fprintf(f, "#define TARGET_X64_AVX\n");
	fprintf(f, "#endif\n");
#elif defined(TARGET_X64_SSE3)
	fprintf(f, "#ifndef TARGET_X64_SSE3\n");
	fprintf(f, "#define TARGET_X64_SSE3\n");
	fprintf(f, "#endif\n");
#elif defined(TARGET_C99_4X4)
	fprintf(f, "#ifndef TARGET_C99_4X4\n");
	fprintf(f, "#define TARGET_C99_4X4\n");
	fprintf(f, "#endif\n");
#elif defined(TARGET_CORTEX_A15)
	fprintf(f, "#ifndef TARGET_CORTEX_A15\n");
	fprintf(f, "#define TARGET_CORTEX_A15\n");
	fprintf(f, "#endif\n");
#elif defined(TARGET_CORTEX_A9)
	fprintf(f, "#ifndef TARGET_CORTEX_A9\n");
	fprintf(f, "#define TARGET_CORTEX_A9\n");
	fprintf(f, "#endif\n");
#endif

    fclose(f);

	return 0;

	}

