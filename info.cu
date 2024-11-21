#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Main Program
int main(void) {
  int device_Count = 0;
  cudaGetDeviceCount(&device_Count);
  if (device_Count == 0) {
    printf("No available CUDA device(s)\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", device_Count);
    for (int i = 0; i < device_Count; i++) {
      cudaDeviceProp device_Prop;
      cudaGetDeviceProperties(&device_Prop, i);
      printf("Device %d: %s\n", i, device_Prop.name);
      printf("  Compute Capability: %d.%d\n", device_Prop.major,
             device_Prop.minor);
      printf("  Total Global Memory: %lu bytes\n", device_Prop.totalGlobalMem);
      printf("  Shared Memory per Block: %lu bytes\n",
             device_Prop.sharedMemPerBlock);
      printf("  Registers per Block: %d\n", device_Prop.regsPerBlock);
      printf("  Warp Size: %d\n", device_Prop.warpSize);
      printf("  Max Threads per Block: %d\n", device_Prop.maxThreadsPerBlock);
      printf("  Max Threads Dimension: (%d, %d, %d)\n",
             device_Prop.maxThreadsDim[0], device_Prop.maxThreadsDim[1],
             device_Prop.maxThreadsDim[2]);
      printf("  Max Grid Size: (%d, %d, %d)\n", device_Prop.maxGridSize[0],
             device_Prop.maxGridSize[1], device_Prop.maxGridSize[2]);
      printf("  Clock Rate: %d\n", device_Prop.clockRate);
      printf("  Total Constant Memory: %lu\n", device_Prop.totalConstMem);
      printf("  Multiprocessor Count(SM): %d\n",
             device_Prop.multiProcessorCount);
      printf("  L2 Cache Size: %d\n", device_Prop.l2CacheSize);
      printf("  Memory Clock Rate: %d\n", device_Prop.memoryClockRate);
      printf("  Memory Bus Width: %d\n", device_Prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth: %f\n",
             2.0 * device_Prop.memoryClockRate *
                 (device_Prop.memoryBusWidth / 8) / 1.0e6);
    }
  }
}
