#include "../src/cuda_target.cuh"
int main() { return validate_cuda_target(true) ? 0 : 1; }
