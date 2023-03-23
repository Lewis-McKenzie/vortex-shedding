CC = nvcc
CFLAGS=-O3
LINK = nvcc
LIBFLAGS=-lm -lcuda -lcudart
NVCCFLAGS =-gencode arch=compute_20,code=sm_20 -m64 -O2

OBJDIR = obj

_OBJ = args.o data.o setup.o vtk.o boundary.o vortex.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

.PHONY: directories

all: directories vortex

obj/%.o: %.cu
	$(CC) -c -o $@ $< $(CFLAGS) 

vortex: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

clean:
	rm -Rf $(OBJDIR)
	rm -f vortex

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

