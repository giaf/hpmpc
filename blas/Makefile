include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = blas_d_avx_lib4.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX -fPIC $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS = blas_d_c99_lib4.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_C99_4X4 -fPIC $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

