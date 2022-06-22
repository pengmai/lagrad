#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/mach_traps.h>
#include <mach/shared_memory_server.h>
#include <mach/task.h>
#include <mach/task_info.h>
#include <mach/thread_act.h>
#include <mach/thread_info.h>
#include <mach/vm_map.h>
#include <mach/vm_region.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/vmmeter.h>

typedef struct vmtotal vmtotal_t;

typedef struct RunProcDyn { /* dynamic process information */
  size_t rss, vsize;
  double utime, stime;
} RunProcDyn;

/* On Mac OS X, the only way to get enough information is to become root. Pretty
 * frustrating!*/
int run_get_dynamic_proc_info(pid_t pid, RunProcDyn *rpd) {
  task_t task;
  kern_return_t error;
  mach_msg_type_number_t count;
  thread_array_t thread_table;
  thread_basic_info_t thi;
  thread_basic_info_data_t thi_data;
  unsigned table_size;
  struct task_basic_info ti;

  error = task_for_pid(mach_task_self(), pid, &task);
  if (error != KERN_SUCCESS) {
    // fprintf(stderr, "++ Probably you have to set suid or become root.\n");
    rpd->rss = rpd->vsize = 0;
    rpd->utime = rpd->stime = 0;
    return 0;
  }
  count = TASK_BASIC_INFO_COUNT;
  error = task_info(task, TASK_BASIC_INFO, (task_info_t)&ti, &count);
  assert(error == KERN_SUCCESS);
  { /* adapted from ps/tasks.c */
    vm_region_basic_info_data_64_t b_info;
    vm_address_t address = GLOBAL_SHARED_TEXT_SEGMENT;
    vm_size_t size;
    mach_port_t object_name;
    count = VM_REGION_BASIC_INFO_COUNT_64;
    error = vm_region_64(task, &address, &size, VM_REGION_BASIC_INFO,
                         (vm_region_info_t)&b_info, &count, &object_name);
    if (error == KERN_SUCCESS) {
      if (b_info.reserved && size == (SHARED_TEXT_REGION_SIZE) &&
          ti.virtual_size >
              (SHARED_TEXT_REGION_SIZE + SHARED_DATA_REGION_SIZE)) {
        ti.virtual_size -= (SHARED_TEXT_REGION_SIZE + SHARED_DATA_REGION_SIZE);
      }
    }
    rpd->rss = ti.resident_size;
    rpd->vsize = ti.virtual_size;
  }
  { /* calculate CPU times, adapted from top/libtop.c */
    unsigned i;
    rpd->utime = ti.user_time.seconds + ti.user_time.microseconds * 1e-6;
    rpd->stime = ti.system_time.seconds + ti.system_time.microseconds * 1e-6;
    error = task_threads(task, &thread_table, &table_size);
    assert(error == KERN_SUCCESS);
    thi = &thi_data;
    for (i = 0; i != table_size; ++i) {
      count = THREAD_BASIC_INFO_COUNT;
      error = thread_info(thread_table[i], THREAD_BASIC_INFO,
                          (thread_info_t)thi, &count);
      assert(error == KERN_SUCCESS);
      if ((thi->flags & TH_FLAGS_IDLE) == 0) {
        rpd->utime +=
            thi->user_time.seconds + thi->user_time.microseconds * 1e-6;
        rpd->stime +=
            thi->system_time.seconds + thi->system_time.microseconds * 1e-6;
      }
      if (task != mach_task_self()) {
        error = mach_port_deallocate(mach_task_self(), thread_table[i]);
        assert(error == KERN_SUCCESS);
      }
    }
    error = vm_deallocate(mach_task_self(), (vm_offset_t)thread_table,
                          table_size * sizeof(thread_array_t));
    assert(error == KERN_SUCCESS);
  }
  mach_port_deallocate(mach_task_self(), task);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Missing pid\n");
    return 1;
  }
  pid_t pid = strtol(argv[1], (char **)NULL, 10);
  RunProcDyn rpd;
  size_t max_rss = 0;
  size_t vsize = 0;
  do {
    usleep(1000);
    run_get_dynamic_proc_info(pid, &rpd);
    // fprintf(stderr, "rss: %zu, vsize: %zu\n", rpd.rss, rpd.vsize);
    if (rpd.rss > max_rss) {
      max_rss = rpd.rss;
      vsize = rpd.vsize;
    }
  } while (rpd.rss > 0);

  printf("max rss: %zu, max vsize: %zu\n", max_rss, vsize);
}
