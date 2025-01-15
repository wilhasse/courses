/* Disable bool redefinition in my_global.h */
#define HAVE_BOOL 1
typedef char my_bool;

#include "my_global.h"
#include "my_sys.h"
#include "mysql/psi/psi.h"

/* Basic PFS implementations */
void 
pfs_register_mutex_v1(const char *category, 
                     PSI_mutex_info_v1 *info,
                     int count) {
}

PSI_mutex_locker* 
pfs_start_mutex_wait_v1(PSI_mutex_locker_state_v1 *state, 
                       PSI_mutex *mutex,
                       PSI_mutex_operation op, 
                       const char *src_file, 
                       uint src_line) {
  return NULL;
}

void pfs_end_mutex_wait_v1(PSI_mutex_locker *locker, int rc) {
}

void pfs_unlock_mutex_v1(PSI_mutex *mutex) {
}

void pfs_memory_alloc_vc(PSI_memory_key key, size_t size, PSI_thread **owner) {
}

void pfs_memory_free_vc(PSI_memory_key key, size_t size, PSI_thread *owner) {
}

void pfs_memory_claim_vc(PSI_memory_key key, size_t size, PSI_thread **owner, my_bool claim) {
}

void pfs_delete_current_thread_vc(void) {
}

void pfs_register_memory_vc(const char *category,
                          PSI_memory_info_v1 *info,
                          int count) {
}

PSI_mutex* pfs_init_mutex_v1(PSI_mutex_key key, const void *identity) {
  return NULL;
}

void pfs_destroy_mutex_v1(PSI_mutex *mutex) {
}

/* File operation stubs */
PSI_file_locker* 
pfs_get_thread_file_name_locker_vc(PSI_file_locker_state_v1 *state,
                                  PSI_file_key key,
                                  PSI_file_operation op,
                                  const char *name,
                                  const void *identity) {
  return NULL;
}

void pfs_start_file_open_wait_vc(PSI_file_locker *locker,
                                const char *src_file,
                                uint src_line) {
}

PSI_file* 
pfs_end_file_open_wait_and_bind_to_descriptor_vc(PSI_file_locker *locker,
                                               File file) {
  return NULL;
}

void pfs_end_file_close_wait_vc(PSI_file_locker *locker, 
                               int rc) {
}

void pfs_start_file_wait_vc(PSI_file_locker *locker,
                           size_t count,
                           const char *src_file,
                           uint src_line) {
}

void pfs_end_file_wait_vc(PSI_file_locker *locker,
                         size_t size) {
}
