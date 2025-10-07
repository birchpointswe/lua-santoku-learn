#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

int main() {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  return 0;
}
