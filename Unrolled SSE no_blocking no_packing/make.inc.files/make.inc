CC = icc
CXX = icpc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISLAB_USE_BLAS),true)
CFLAGS = -$(COMPILER_OPT_LEVEL) -std=c99  -mavx -fPIC -DUSE_BLAS
else
CFLAGS = -$(COMPILER_OPT_LEVEL) -std=c99  -mavx -fPIC
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/libblislab.so

LDFLAGS = -I$(BLISLAB_DIR)/include -I$(BLISLAB_DIR)/kernels -I/$(BLISLAB_MKL_DIR)/include

LDLIBS = $(LIBBLISLAB) -lpthread -lm -mkl=parallel -Werror -Wall -pedantic

OS_NAME = $(shell uname -s)
ifeq ($(OS_NAME),Linux)
LDLIBS += -lrt
endif

