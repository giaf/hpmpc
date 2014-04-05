include ../Makefile.rule

CODEGEN_OBJS = code_generator_blas_d_avx_lib4.o code_generator_dricposv.o
OBJS = dricposv_codegen.o

ifeq ($(TARGET), AVX)
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif

codegen: $(CODEGEN_OBJS)
	gcc -o code_generator.out $(CODEGEN_OBJS) -O2
	./code_generator.out

obj: codegen $(OBJS)

clean:
	rm -f $(OBJS)
	rm -f code_generator.out

code_generator_dricposv.o: code_generator_dricposv.c ../problem_size.h
code_generator_blas_d_avx_lib4.o: code_generator_blas_d_avx_lib4.c ../problem_size.h
