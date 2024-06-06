# Compiler
CC = gcc

# Compiler flags
CFLAGS = -fPIC -Wall -Werror

# Linker flags
LDFLAGS = -shared

# Target
TARGET = libzero.so

# Source files
SRCS = libzero.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default rule
all: $(TARGET)

# Rule to create the shared library
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Rule to create object files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)
