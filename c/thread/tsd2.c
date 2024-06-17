#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

__thread int counter = 0;

// Function prototype declaration
void* increment(void* arg);

void* run(void* arg) {
    increment(NULL);
    increment(NULL);
    increment(NULL);
    return NULL;
}

void* increment(void* arg) {
    counter += 1;
    printf("Thread %ld, Counter: %d\n", pthread_self(), counter);
    return NULL;
}

int main() {
    pthread_t tid1;

    pthread_create(&tid1, NULL, run, NULL);
    pthread_join(tid1, NULL);

    return 0;
}
