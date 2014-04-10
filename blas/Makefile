include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = blas_d_avx_lib4.o
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX -fPIC $(DEBUG)
endif
ifeq ($(TARGET), NEON)
OBJS = blas_d_neon_lib4.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), POWERPC_G2)
OBJS = blas_d_ppc_lib4.o
CFLAGS = $(OPT) -std=c99 -fPIC -mcpu=603e -DTARGET_POWERPC_G2 $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS = blas_d_c99_lib4.o
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 -fPIC $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
OBJS = blas_d_c99_lib2.o
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_2X2 -fPIC $(DEBUG)
endif
ifeq ($(TARGET), ATOM)
OBJS = blas_d_atom_lib2.o
CFLAGS = $(OPT) -std=c99 -msse3 -mfpmath=sse -march=atom -DTARGET_ATOM -fPIC $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

