#include <pthread.h>
#include <stdio.h>

#define THREAD_COUNT 10

// compile: gcc -o mutex mutex.c -pthread
int counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

// Thread function to execute.
void *thread_function(void *arg) {

    for (int i = 0;i < 10000000; i++) {

	pthread_mutex_lock(&lock);
	counter++;
	pthread_mutex_unlock(&lock);
    }
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
    pthread_mutex_destroy(&lock);

    return 0;
}
