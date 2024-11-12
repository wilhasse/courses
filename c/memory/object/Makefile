# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -Wno-unused-but-set-parameter -g -I../munit

# Source files
LOCAL_SOURCES = snekobject.c main.c
MUNIT_SOURCES = ../munit/munit.c
SOURCES = $(LOCAL_SOURCES) $(MUNIT_SOURCES)

# Headers
LOCAL_HEADERS = snekobject.h
MUNIT_HEADERS = ../munit/munit.h
HEADERS = $(LOCAL_HEADERS) $(MUNIT_HEADERS)

OBJECTS = $(SOURCES:.c=.o)
TARGET = test_exercise

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) -lm

# Compile source files to object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Special rule for munit.o since it's in a different directory
../munit/munit.o: ../munit/munit.c ../munit/munit.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f ../munit/*.o

# Run the tests
test: $(TARGET)
	./$(TARGET)

# Phony targets
