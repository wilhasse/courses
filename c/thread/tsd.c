#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

__thread int tls_var = 0;

void* thread_function(void *arg) {
    tls_var = (int)(long)arg;
    printf("Thread %ld has TLS variable %d\n", pthread_self(), tls_var);
    sleep(1);
    return NULL;
}

int main() {
    pthread_t tid1, tid2;

    pthread_create(&tid1, NULL, thread_function, (void*)1);
    pthread_create(&tid2, NULL, thread_function, (void*)2);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);


    return 0;
}
