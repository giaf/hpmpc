include ../Makefile.rule

ifeq ($(TARGET), AVX)
OBJS = dricposv.o
CFLAGS = $(OPT) -std=c99 -mavx -fPIC -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), NEON)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), POWERPC_G2)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -fPIC -mcpu=603e -DTARGET_POWERPC_G2 $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_4X4 $(DEBUG)
endif
ifeq ($(TARGET), C99_2X2)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -fPIC -DTARGET_C99_2X2 $(DEBUG)
endif
ifeq ($(TARGET), ATOM)
OBJS = dricposv.o sricposv.o
CFLAGS = $(OPT) -std=c99 -msse3 -mfpmath=sse -march=atom -fPIC -DTARGET_ATOM $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o

