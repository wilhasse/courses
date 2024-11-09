#include <string.h>

typedef struct {
  char buffer[64];
  size_t length;
} TextBuffer;

int smart_append(TextBuffer* dest, const char* src);
