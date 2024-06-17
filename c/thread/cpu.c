#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

void* thread_function(void* arg);

void set_cpu_affinity() {
    pthread_t thread;
    pthread_attr_t attr;
    cpu_set_t cpus;

    pthread_attr_init(&attr);
    CPU_ZERO(&cpus);
    CPU_SET(0, &cpus);  // Set thread to run on CPU 0

    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
    pthread_create(&thread, &attr, thread_function, NULL);
    pthread_join(thread, NULL);
    pthread_attr_destroy(&attr);
}

void* thread_function(void* arg) {
    // Thread tasks here
    return NULL;
}
int main() {

    set_cpu_affinity();
}
