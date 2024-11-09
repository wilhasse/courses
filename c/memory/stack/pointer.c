#include <stdio.h>
#include <stdlib.h>

/*
 Correct code using pointer and allocating memory dynamically
*/
typedef struct {
    int x;
    int y;
} coord_t;

coord_t* new_coord(int x, int y) {
    // Allocate memory for the coordinate
    coord_t* c = (coord_t*)malloc(sizeof(coord_t));
    if (c == NULL) {
        // Handle allocation failure
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize the coordinate
    c->x = x;
    c->y = y;
    return c;
}

// Helper function to free the coordinate
void free_coord(coord_t* c) {
    free(c);
}

int main() {
    // Create coordinates using pointers
    coord_t* c1 = new_coord(10, 20);
    coord_t* c2 = new_coord(30, 40);
    coord_t* c3 = new_coord(50, 60);

    // Print the coordinates
    printf("c1: %d, %d\n", c1->x, c1->y);
    printf("c2: %d, %d\n", c2->x, c2->y);
    printf("c3: %d, %d\n", c3->x, c3->y);

    // Don't forget to free the memory
    free_coord(c1);
    free_coord(c2);
    free_coord(c3);

    return 0;
}
