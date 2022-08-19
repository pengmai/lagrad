#include <stdlib.h>

typedef struct vmtotal vmtotal_t;

typedef struct RunProcDyn { /* dynamic process information */
  size_t rss, vsize;
  double utime, stime;
} RunProcDyn;

/* On Mac OS X, the only way to get enough information is to become root. Pretty
 * frustrating!*/
int run_get_dynamic_proc_info(pid_t pid, RunProcDyn *rpd);
