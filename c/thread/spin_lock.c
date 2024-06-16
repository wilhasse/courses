#include <pthread.h>
#include <stdio.h>

pthread_spinlock_t spinlock;
int counter = 0;

void* increment_counter(void* arg) {
    for (int i = 0; i < 1000; i++) {
        pthread_spin_lock(&spinlock);
        counter++;
        pthread_spin_unlock(&spinlock);
    }
    return NULL;
}

int main() {
    pthread_spin_init(&spinlock, 0);
    pthread_t t1, t2;
    pthread_create(&t1, NULL, increment_counter, NULL);
    pthread_create(&t2, NULL, increment_counter, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Final counter value: %d\n", counter);
    pthread_spin_destroy(&spinlock);
    return 0;
}
