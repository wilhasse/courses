# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra
INCLUDES = -I.
LDFLAGS = -L/usr/local/lib
LIBS = -linnodb -lpthread -lrt -ldl

# Target executables
TARGETS = create_db test_data

# Default target
all: $(TARGETS)

# Rule for building the executables
create_db: create_db.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

test_data: test_data.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

# Clean target
clean:
	rm -f $(TARGETS) *.o

# Prevent make from confusing the file names with target names
.PHONY: all clean
