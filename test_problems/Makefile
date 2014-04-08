include ../Makefile.rule

LIBS = -lm #-lopenblas

#OBJS_TEST = test_dgemm_orig.o
#OBJS_TEST = test_dgemm.o
OBJS_TEST = tools.o test_dricposv.o
#OBJS_TEST = tools.o test_ip_box_lib.o

ifeq ($(TARGET), AVX)
CFLAGS = $(OPT) -std=c99 -mavx -DTARGET_AVX $(DEBUG)
endif
ifeq ($(TARGET), NEON)
CFLAGS = $(OPT) -std=c99 -fPIC -marm -mfloat-abi=softfp -mfpu=neon -mcpu=cortex-a9 -DTARGET_NEON $(DEBUG)
endif
ifeq ($(TARGET), C99_4X4)
CFLAGS = $(OPT) -std=c99 -DTARGET_C99_4X4 $(DEBUG)
endif

obj: $(OBJS_TEST)
	$(CC) -o test.out $(OBJS_TEST) -L. HPMPC.a $(LIBS) #-pg

clean:
	rm -f *.o
	rm -f test.out
	rm -f HPMPC.a

test_dricposv.o: test_dricposv.c test_param.h problem_size.h
test_ip_box_lib.o: test_ip_box_lib.c test_param.h problem_size.h
