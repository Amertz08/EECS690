/*
    Normalizes sum of index to max sum value found
*/
__kernel
void SumKernel(int cols, float maxSum, __global const float* workSum, __global unsigned char* sumArr)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = cols * x + y;
    sumArr[idx] = (unsigned char)round((workSum[idx] / maxSum) * 255.0);
}
