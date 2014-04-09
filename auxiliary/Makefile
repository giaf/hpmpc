include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = aux_d_c99.o aux_s_c99.o block_size_avx.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), NEON)
OBJS = aux_d_c99.o aux_s_c99.o block_size_neon.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS = aux_d_c99.o aux_s_c99.o block_size_c99_4x4.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_4X4 $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
OBJS = aux_d_c99.o aux_s_c99.o block_size_c99_2x2.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_2X2 $(DEBUG)
endif
ifeq ($(TARGET), ATOM)
OBJS = aux_d_c99.o aux_s_c99.o block_size_atom.o
CFLAGS = $(OPT) -std=c99 -fPIC -msse3 -mfpmath=sse -march=atom -DTARGET_ATOM $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o
