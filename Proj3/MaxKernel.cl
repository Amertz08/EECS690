
__kernel
void MaxKernel(int depth, int cols, int rows, int projType, __global const unsigned char* img, __global unsigned char* maxArr, __global float* workSum)
{
    int x, y, z, rowCount, colCount, maxOffset;
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    // Calculate max and sum
    int max = 0;
    float sum = 0.00;
    for (int i = 0; i < depth; i++) { // TODO: might have z value incorrect
        // Map x', y', z', rows', and cols' to calculate index;
        switch (projType) {
            case 1:
                x = idx;
                y = idy;
                z = i;
                colCount = cols;
                rowCount = rows;
                break;
            case 2:
                x = (cols - idx);
                y = idy;
                z = (depth - i);
                colCount = cols;
                rowCount = rows;
                break;
            case 3:
                x = i;
                y = idy;
                z = (cols - idx);
                colCount = depth;
                rowCount = rows;
                break;
            case 4:
                x = (depth - i);
                y = idy;
                z = idx;
                colCount = depth;
                rowCount = rows;
                break;
            case 5:
                x = idx;
                y = i;
                z = (rows - idy);
                colCount = cols;
                rowCount = depth;
                break;
            case 6:
                x = idx;
                y = (depth - i);
                z = idy;
                colCount = cols;
                rowCount = depth;
                break;
            default: // TODO: exit Kernel?
                break;
        }
        // Calculate the offset and get the value
        int offset = y + rowCount * (z * colCount + x);
        int val = (int)img[offset];

        // See if value is max
        if (val > max) {
            max = val;
            maxOffset = x * y;
        }
        // sum += ((i + 1) / depth) * val;
    }

    maxArr[maxOffset] = (unsigned char)max;
//    workSum[col * row] = sum;
}
