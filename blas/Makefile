include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = blas_d_avx_lib4.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX -fPIC $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

