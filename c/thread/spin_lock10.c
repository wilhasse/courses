#include <pthread.h>
#include <stdio.h>

#define THREAD_COUNT 10

// compile: gcc -o mutex mutex.c -pthread
int counter = 0;
pthread_spinlock_t spinlock;

// Thread function to execute.
void *thread_function(void *arg) {

    for (int i = 0;i < 10000000; i++) {

	pthread_spin_lock(&spinlock);
	counter++;
	pthread_spin_unlock(&spinlock);
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
    pthread_spin_destroy(&spinlock);

    return 0;
}
