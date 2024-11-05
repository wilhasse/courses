# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -g

# Source files
SOURCES = exercise.c munit.c main.c
HEADERS = exercise.h munit.h
OBJECTS = $(SOURCES:.c=.o)
TARGET = test_runner

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) -lm

# Compile source files to object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the tests
test: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean test
