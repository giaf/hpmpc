###################################################################################################
#                                                                                                 #
# This file is part of HPMPC.                                                                     #
#                                                                                                 #
# HPMPC -- Library for High-Performance implementation of solvers for MPC.                        #
# Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     #
#                                                                                                 #
# HPMPC is free software; you can redistribute it and/or                                          #
# modify it under the terms of the GNU Lesser General Public                                      #
# License as published by the Free Software Foundation; either                                    #
# version 2.1 of the License, or (at your option) any later version.                              #
#                                                                                                 #
# HPMPC is distributed in the hope that it will be useful,                                        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            #
# See the GNU Lesser General Public License for more details.                                     #
#                                                                                                 #
# You should have received a copy of the GNU Lesser General Public                                #
# License along with HPMPC; if not, write to the Free Software                                    #
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  #
#                                                                                                 #
# Author: Gianluca Frison, giaf (at) dtu.dk                                                       #
#                                                                                                 #
###################################################################################################

include ../Makefile.rule

OBJS = d_ric_sv_codegen.o d_res_codegen.o s_ric_sv_codegen.o s_res_codegen.o

CODEGEN_OBJS_DOUBLE = code_generator_blas_d_lib4.o code_generator_d_ric_sv.o
CODEGEN_OBJS_SINGLE = code_generator_blas_s_lib4.o code_generator_s_ric_sv.o

codegen: $(CODEGEN_OBJS_DOUBLE) $(CODEGEN_OBJS_SINGLE)
	gcc -o code_generator.out $(CODEGEN_OBJS_DOUBLE) -O2
	./code_generator.out
	gcc -o code_generator.out $(CODEGEN_OBJS_SINGLE) -O2
	./code_generator.out
#	sleep 5

obj: codegen $(OBJS)

clean:
	rm -f d_ric_sv_codegen.c 
	rm -f s_ric_sv_codegen.c
	rm -f d_res_codegen.c 
	rm -f s_res_codegen.c
	rm -f *.o
	rm -f code_generator.out

code_generator_d_ric_sv.o: code_generator_d_ric_sv.c ../problem_size.h
code_generator_s_ric_sv.o: code_generator_d_ric_sv.c ../problem_size.h
code_generator_blas_s_lib8.o: code_generator_blas_s_lib8.c ../problem_size.h
code_generator_blas_d_lib4.o: code_generator_blas_d_lib4.c ../problem_size.h
code_generator_blas_s_lib4.o: code_generator_blas_s_lib4.c ../problem_size.h
code_generator_blas_d_lib2.o: code_generator_blas_d_lib2.c ../problem_size.h
code_generator_blas_s_lib2.o: code_generator_blas_s_lib2.c ../problem_size.h

