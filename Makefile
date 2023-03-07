CC=gcc
CFLAGS=-O3 
LIBFLAGS=-lm

OBJDIR = obj

_OBJ = args.o data.o setup.o vtk.o boundary.o vortex.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

.PHONY: directories

all: directories vortex

obj/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) 

vortex: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

clean:
	rm -Rf $(OBJDIR)
	rm -f vortex

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

