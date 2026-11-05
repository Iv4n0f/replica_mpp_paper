import os

# 1. core_c.h
path = '/home/alumno11/src/task-bench/core/core_c.h'
with open(path, 'r') as f:
    c = f.read()

c = c.replace(
'''  LOAD_IMBALANCE,
} kernel_type_t;''',
'''  LOAD_IMBALANCE,
  COMPUTE_BOUND_GPU,
} kernel_type_t;''')

c = c.replace(
'''  double
      imbalance;  // amount of imbalance as a fraction of the number of iterations
} kernel_t;''',
'''  double
      imbalance;  // amount of imbalance as a fraction of the number of iterations
  long nblocks;
  long tpblock;
} kernel_t;''')

with open(path, 'w') as f:
    f.write(c)
print('Patched core_c.h')

# 2. core_kernel.h
path = '/home/alumno11/src/task-bench/core/core_kernel.h'
with open(path, 'r') as f:
    c = f.read()

c = c.replace(
'''double execute_kernel_imbalance(const Kernel &kernel, long graph_index,
                                long timestep, long point);''',
'''double execute_kernel_imbalance(const Kernel &kernel, long graph_index,
                                long timestep, long point);

void execute_kernel_compute_bound_gpu(const Kernel &kernel, long point);''')

with open(path, 'w') as f:
    f.write(c)
print('Patched core_kernel.h')

# 3. core_kernel.cc
path = '/home/alumno11/src/task-bench/core/core_kernel.cc'
with open(path, 'r') as f:
    c = f.read()

gpu_kernel_code = '''
#include <omp.h>

void execute_kernel_compute_bound_gpu(const Kernel &kernel, long point)
{
  int num_devices = omp_get_num_devices();
  int device_id = (num_devices > 0) ? (int)(point % num_devices) : 0;
  long nblocks = kernel.nblocks > 0 ? kernel.nblocks : 1024;
  long tpblock = kernel.tpblock > 0 ? kernel.tpblock : 512;
  long iterations = kernel.iterations;
  long total_threads = nblocks * tpblock;

  float sink = 0.0f;
  #pragma omp target teams distribute parallel for num_teams(nblocks) thread_limit(tpblock) device(device_id) map(tofrom: sink)
  for (long i = 0; i < total_threads; ++i) {
    float a = 1.1f, b = 1.2f, c = 1.3f + (float)(i % 10);
    #pragma unroll 16
    for (long iter = 0; iter < iterations; ++iter) {
      c = a * b + c;
      a = b * c + a;
      b = c * a + b;
      c = a * b + c;
      a = b * c + a;
      b = c * a + b;
      c = a * b + c;
      a = b * c + a;
      b = c * a + b;
      c = a * b + c;
      a = b * c + a;
      b = c * a + b;
      c = a * b + c;
      a = b * c + a;
      b = c * a + b;
      c = a * b + c;
    }
    if (c == 0.1234567f) sink += c;
  }
}
'''

c += gpu_kernel_code
with open(path, 'w') as f:
    f.write(c)
print('Patched core_kernel.cc')

# 4. core.cc
path = '/home/alumno11/src/task-bench/core/core.cc'
with open(path, 'r') as f:
    c = f.read()

# execute switch
c = c.replace(
'''  case KernelType::LOAD_IMBALANCE:
    assert(timestep >= 0 && point >= 0);
    execute_kernel_imbalance(*this, graph_index, timestep, point);
    break;''',
'''  case KernelType::LOAD_IMBALANCE:
    assert(timestep >= 0 && point >= 0);
    execute_kernel_imbalance(*this, graph_index, timestep, point);
    break;
  case KernelType::COMPUTE_BOUND_GPU:
    execute_kernel_compute_bound_gpu(*this, point);
    break;''')

# ktype_by_name
c = c.replace(
'''    {"load_imbalance", KernelType::LOAD_IMBALANCE},
};''',
'''    {"load_imbalance", KernelType::LOAD_IMBALANCE},
    {"compute_bound_gpu", KernelType::COMPUTE_BOUND_GPU},
};''')

# flags
c = c.replace(
'#define IMBALANCE_FLAG "-imbalance"',
'''#define IMBALANCE_FLAG "-imbalance"
#define NBLOCKS_FLAG "-nblocks"
#define TPBLOCK_FLAG "-tpblock"''')

# CLI parsing
cli_target = '''    if (!strcmp(argv[i], IMBALANCE_FLAG)) {
      needs_argument(i, argc, IMBALANCE_FLAG);
      double value = atof(argv[++i]);
      if (value < 0 || value > 2) {
        fprintf(stderr,
                "error: Invalid flag \\"" IMBALANCE_FLAG
                " %f\\" must be >= 0 and <= 2\\n",
                value);
        abort();
      }
      graph.kernel.imbalance = value;
    }'''

cli_replacement = cli_target + '''

    if (!strcmp(argv[i], NBLOCKS_FLAG)) {
      needs_argument(i, argc, NBLOCKS_FLAG);
      long value = atol(argv[++i]);
      if (value <= 0) {
        fprintf(stderr,
                "error: Invalid flag \\"" NBLOCKS_FLAG " %ld\\" must be > 0\\n",
                value);
        abort();
      }
      graph.kernel.nblocks = value;
    }

    if (!strcmp(argv[i], TPBLOCK_FLAG)) {
      needs_argument(i, argc, TPBLOCK_FLAG);
      long value = atol(argv[++i]);
      if (value <= 0) {
        fprintf(stderr,
                "error: Invalid flag \\"" TPBLOCK_FLAG " %ld\\" must be > 0\\n",
                value);
        abort();
      }
      graph.kernel.tpblock = value;
    }'''

c = c.replace(cli_target, cli_replacement)

# default_graph
c = c.replace(
'  graph.kernel = {KernelType::EMPTY, 0, 16, 0.0};',
'  graph.kernel = {KernelType::EMPTY, 0, 16, 0.0, 1024, 512};')

# count_flops_per_task
c = c.replace(
'''  case KernelType::LOAD_IMBALANCE:
    {
      long iterations =
          select_imbalance_iterations(g.kernel, g.graph_index, timestep, point);
      return 2 * 64 * iterations + 64;
    }''',
'''  case KernelType::LOAD_IMBALANCE:
    {
      long iterations =
          select_imbalance_iterations(g.kernel, g.graph_index, timestep, point);
      return 2 * 64 * iterations + 64;
    }

  case KernelType::COMPUTE_BOUND_GPU:
    return (long long)g.kernel.nblocks * g.kernel.tpblock * g.kernel.iterations * 32;''')

# count_bytes_per_task
c = c.replace(
'''  case KernelType::LOAD_IMBALANCE:
    return 0;''',
'''  case KernelType::LOAD_IMBALANCE:
  case KernelType::COMPUTE_BOUND_GPU:
    return 0;''')

with open(path, 'w') as f:
    f.write(c)
print('Patched core.cc')
