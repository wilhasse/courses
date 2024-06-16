#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

sem_t semaphore;
char message[100];  // Shared resource

// Publisher thread function
void* publisher(void* arg) {
    sleep(2);
    sprintf(message, "Data published");
    sem_post(&semaphore);  // Signal subscriber
    return NULL;
}

// Subscriber thread function
void* subscriber(void* arg) {
    sem_wait(&semaphore);  // Wait for the signal
    printf("Received message: %s\n", message);
    return NULL;
}

int main() {
    pthread_t pub, sub;
    sem_init(&semaphore, 0, 0);
    pthread_create(&pub, NULL, publisher, NULL);
    pthread_create(&sub, NULL, subscriber, NULL);
    pthread_join(pub, NULL);
    pthread_join(sub, NULL);
    sem_destroy(&semaphore);
    return 0;
}
