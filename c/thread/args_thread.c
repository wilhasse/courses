#include <pthread.h>
#include <stdio.h>

#define THREAD_COUNT 10

typedef struct {
	int arg1;
	int arg2;
} thread_arg_t;

// Thread function to execute.
void *thread_function(void *vargs) {
    thread_arg_t *args = (thread_arg_t*)vargs;
    printf("Hello from the thread %d!\n", args->arg1);
    return NULL;
}

int main(int argc, char *argv) {
    pthread_t threads[THREAD_COUNT];
    thread_arg_t myargs;

    // Create the thread
    int i = 0;
    for (i = 0; i < THREAD_COUNT; i++) {
	   myargs.arg1 = i;
 	   if (pthread_create(&threads[i], NULL, thread_function, (void*) &myargs)) {
        	perror("phread_create");
	        return -1;
	    }

    }

    for (i = 0; i < THREAD_COUNT; i++) {
    	pthread_join(threads[i], NULL);
    }

    return 0;
}
