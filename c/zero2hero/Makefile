# Define the compiler
CC = gcc

# Define compiler flags
CFLAGS = -Wall -g

# Define the target executable
TARGET = hello

# Define the source files
SRCS = hello.c string.c union.c struct.c

# Define the object files
OBJS = $(SRCS:.c=.o)

# Default rule
all: $(TARGET)

# Rule to link the object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile the source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule to remove generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets to avoid conflicts with files of the same name
.PHONY: all clean
