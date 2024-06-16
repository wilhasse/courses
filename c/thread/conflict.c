#include <pthread.h>
#include <stdio.h>

#define THREAD_COUNT 10

// compile: gcc -o conflict conflict.c -pthread
int counter = 0;

// Thread function to execute.
void *thread_function(void *arg) {

    for (int i = 0;i < 1000000; i++) {

	counter++;
    }
    //printf("Counter is %d\n",counter);
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

    printf("Final counter value: %d\n", counter);

    return 0;
}
