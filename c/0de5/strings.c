#include <stdio.h>

// allocate buffer on stack memory
// it doesn't live when function returns
char *my_wrong_strcat(char *a, char *b) {

    char buffer[5] = {0};
    sprintf(buffer, "%s%s", a, b);
    return buffer;
}

// data section
void my_strcat(char *buffer,char *a, char *b) {

    sprintf(buffer, "%s%s", a, b);
    return buffer;
}

// allocate in heap
// problem no ono is freeing the result (malloc)
char *my_malloc_strcat(char a[], char b[]) {

    char *result = malloc((strlen(a) + strlen(b) + 1) * sizeof(char));
    sprintf(result, "%s%s", a, b);
    return result;
}

int main() {

    char result[5] = {0};
    char *result2;
    char *result3;

    my_strcat(result,"hi","ya");
    printf("result: %s\n", result);

    result2 = my_wrong_strcat("hi","ya");
    printf("result2: %s\n", result2);

    result3 = my_malloc_strcat("hi","ya");
    printf("result3: %s\n", result3);
}
