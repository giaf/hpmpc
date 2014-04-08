include ../Makefile.rule

OBJS = ip_d_box.o

ifeq ($(TARGET), AVX)
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), NEON)
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 $(DEBUG)
endif

obj: $(OBJS)

clean:
	rm -f *.o


