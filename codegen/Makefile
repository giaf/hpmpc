include ../Makefile.rule

OBJS = dricposv_codegen.o sricposv_codegen.o

ifeq ($(TARGET), AVX)
CODEGEN_OBJS = code_generator_blas_d_avx_lib4.o code_generator_dricposv.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), NEON)
CODEGEN_OBJS = code_generator_blas_d_neon_lib4.o code_generator_dricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), POWERPC_G2)
CODEGEN_OBJS = code_generator_blas_d_ppc_lib4.o code_generator_dricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -mcpu=603e -DTARGET_POWERPC_G2 $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
CODEGEN_OBJS_DOUBLE = code_generator_blas_d_c99_4x4_lib4.o code_generator_dricposv.o
CODEGEN_OBJS_SINGLE = code_generator_blas_s_c99_4x4_lib4.o code_generator_sricposv.o
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
CODEGEN_OBJS = code_generator_blas_d_c99_2x2_lib2.o code_generator_dricposv.o
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_2X2 $(DEBUG)
endif
ifeq ($(TARGET), ATOM)
CODEGEN_OBJS = code_generator_blas_d_atom_lib2.o code_generator_dricposv.o
CFLAGS = $(OPT) -std=c99 -msse3 -mfpmath=sse -march=atom -DTARGET_ATOM $(DEBUG)
endif

codegen: $(CODEGEN_OBJS_DOUBLE) $(CODEGEN_OBJS_SINGLE)
	gcc -o code_generator.out $(CODEGEN_OBJS_DOUBLE) -O2
	./code_generator.out
	gcc -o code_generator.out $(CODEGEN_OBJS_SINGLE) -O2
	./code_generator.out

obj: codegen $(OBJS)

clean:
	rm -f *.o
	rm -f code_generator.out

code_generator_dricposv.o: code_generator_dricposv.c ../problem_size.h
code_generator_sricposv.o: code_generator_dricposv.c ../problem_size.h
code_generator_blas_d_avx_lib4.o: code_generator_blas_d_avx_lib4.c ../problem_size.h
code_generator_blas_d_neon_lib4.o: code_generator_blas_d_neon_lib4.c ../problem_size.h
code_generator_blas_d_ppc_lib4.o: code_generator_blas_d_ppc_lib4.c ../problem_size.h
code_generator_blas_d_c99_4x4_lib4.o: code_generator_blas_d_c99_4x4_lib4.c ../problem_size.h
code_generator_blas_s_c99_4x4_lib4.o: code_generator_blas_s_c99_4x4_lib4.c ../problem_size.h
code_generator_blas_d_c99_2x2_lib2.o: code_generator_blas_d_c99_2x2_lib2.c ../problem_size.h
