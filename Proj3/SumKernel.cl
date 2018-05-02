
__kernel
void SumKernel(size_t depth, size_t x, size_t y, float maxSum, __global const float* workSum, __global const int* sumArr)
{
    auto col = get_global_id(0);
    auto row = get_global_id(1);

    // TODO: proper index calculations
    for (int i = 0; i < depth; i++) {
        auto idx = col * row * i;
        sumArr[idx] = round((workSum[idx] / maxSum) * 255.0);
    }
}