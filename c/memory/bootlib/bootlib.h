// bootlib.h
#ifndef BOOTLIB_H
#define BOOTLIB_H

#include <stdbool.h>
#include <stdlib.h>

void* tracked_malloc(size_t size, const char* file, int line);
void tracked_free(void* ptr, const char* file, int line);
bool boot_all_freed(void);
bool boot_is_freed(void* ptr);

// Undefine malloc/free first in case they were already defined
#ifdef malloc
#undef malloc
#endif

#ifdef free
#undef free
#endif

// Override malloc and free with tracked versions
#define malloc(size) tracked_malloc(size, __FILE__, __LINE__)
#define free(ptr) tracked_free(ptr, __FILE__, __LINE__)

#endif
