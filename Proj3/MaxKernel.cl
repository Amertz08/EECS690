
__kernel
void MaxKernel(int depth, int cols, int rows, int projType, __global const unsigned char* img, __global unsigned char* maxArr, __global float* workSum)
{
    int x, y, z, rowCount, colCount, sheetCount, maxOffset;
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    switch (projType) {
        case 1:
        case 2:
            colCount = cols;
            rowCount = rows;
            sheetCount = depth;
            break;
        case 3:
        case 4:
            colCount = depth;
            rowCount = rows;
            sheetCount = cols;
            break;
        case 5:
        case 6:
            colCount = cols;
            rowCount = depth;
            sheetCount = rows;
            break;
    }

    // Calculate max and sum
    int max = 0;
    float sum = 0.00;
    for (int i = 0; i < sheetCount; i++) {
        // Map x', y', z', rows', and cols' to calculate index;
        switch (projType) {
            case 1:
                x = idx;
                y = idy;
                z = i;
                break;
            case 2:
                x = (cols - idx);
                y = idy;
                z = (depth - i);
                break;
            case 3:
                x = i;
                y = idy;
                z = (cols - idx);
                break;
            case 4:
                x = (depth - i);
                y = idy;
                z = idx;
                break;
            case 5:
                x = idx;
                y = i;
                z = (rows - idy);
                break;
            case 6:
                x = idx;
                y = (depth - i);
                z = idy;
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
