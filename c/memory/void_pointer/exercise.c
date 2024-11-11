#include "exercise.h"
#include <stdbool.h>
#include <stdio.h>

bool snek_zero_out(void *ptr, snek_object_kind_t kind) {
    if (ptr == NULL) {
        return false;
    }

    switch (kind) {
        case INTEGER:
            ((snek_int_t*)ptr)->value = 0;
            break;
        case FLOAT:
            ((snek_float_t*)ptr)->value = 0.0f;
            break;
        case BOOL:
            ((snek_bool_t*)ptr)->value = false;
            break;
        default:
            return false;
    }
    return true;
}
