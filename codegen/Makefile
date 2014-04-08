include ../Makefile.rule

OBJS = dricposv_codegen.o

ifeq ($(TARGET), AVX)
CODEGEN_OBJS = code_generator_blas_d_avx_lib4.o code_generator_dricposv.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), NEON)
CODEGEN_OBJS = code_generator_blas_d_neon_lib4.o code_generator_dricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
CODEGEN_OBJS = code_generator_blas_d_c99_4x4_lib4.o code_generator_dricposv.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_C99_4X4 $(DEBUG)
endif

codegen: $(CODEGEN_OBJS)
	gcc -o code_generator.out $(CODEGEN_OBJS) -O2
	./code_generator.out

obj: codegen $(OBJS)

clean:
	rm -f *.o
	rm -f code_generator.out

code_generator_dricposv.o: code_generator_dricposv.c ../problem_size.h
code_generator_blas_d_avx_lib4.o: code_generator_blas_d_avx_lib4.c ../problem_size.h
code_generator_blas_d_neon_lib4.o: code_generator_blas_d_neon_lib4.c ../problem_size.h
code_generator_blas_d_c99_4x4_lib4.o: code_generator_blas_d_c99_4x4_lib4.c ../problem_size.h
