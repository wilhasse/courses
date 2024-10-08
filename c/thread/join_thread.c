#include <pthread.h>
#include <stdio.h>

#define THREAD_COUNT 10

// compile: gcc -o join_thread join_thread.c -pthread

// Thread function to execute.
void *thread_function(void *arg) {
    printf("Hello from the thread!\n");
    return NULL;
}

int main() {
    pthread_t threads[THREAD_COUNT];

    // Create the thread
    int i = 0;
    for (i = 0; i < THREAD_COUNT; i++) {
 	   if (pthread_create(&threads[i], NULL, thread_function, NULL)) {
        	perror("phread_create");
	        return -1;
	    }
    }

    for (i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
