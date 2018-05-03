#include <iostream>
#include <fstream>
#include <string>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>

#else
    #include <CL/opencl.h>
#endif

#define DEBUG true

#include "ImageWriter.h"

#include "helpers.hpp"

auto devType = CL_DEVICE_TYPE_ALL;

void print_platforms(cl_platform_id* p, int count)
{
    for (int i = 0; i < count; i++) {
        reportPlatformInformation(p[i]);
    }
}

/**
 * Modified from https://bit.ly/2r2DriH
 * @param devices
 * @param deviceCount
 */
void print_devices(cl_device_id* devices, int deviceCount)
{
    for (int i = 0; i < deviceCount; i++) {
        char device_string[1024];
        std::cout << "Device index: " << i << std::endl;
        // CL_DEVICE_NAME
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
        printf("CL_DEVICE_NAME: \t\t\t%s\n", device_string);

        // CL_DEVICE_VENDOR
        clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
        printf("CL_DEVICE_VENDOR: \t\t\t%s\n", device_string);

        // CL_DRIVER_VERSION
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
        printf("CL_DRIVER_VERSION: \t\t\t%s\n", device_string);

        // CL_DEVICE_INFO
        cl_device_type type;
        clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
        if( type & CL_DEVICE_TYPE_CPU )
            printf("CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_CPU");
        if( type & CL_DEVICE_TYPE_GPU )
            printf("CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
        if( type & CL_DEVICE_TYPE_ACCELERATOR )
            printf("CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_ACCELERATOR");
        if( type & CL_DEVICE_TYPE_DEFAULT )
            printf("CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_DEFAULT");

        // CL_DEVICE_MAX_COMPUTE_UNITS
        cl_uint compute_units;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
        printf("CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", compute_units);

        // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
        size_t workitem_dims;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
        printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u\n", workitem_dims);

        // CL_DEVICE_MAX_WORK_ITEM_SIZES
        size_t workitem_size[3];
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
        printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u \n", workitem_size[0], workitem_size[1], workitem_size[2]);

        // CL_DEVICE_MAX_WORK_GROUP_SIZE
        size_t workgroup_size;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
        printf("CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%u\n", workgroup_size);

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        cl_uint clock_frequency;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
        printf("CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz\n", clock_frequency);

        // CL_DEVICE_ADDRESS_BITS
        cl_uint addr_bits;
        clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
        printf("CL_DEVICE_ADDRESS_BITS:\t\t%u\n", addr_bits);

        // CL_DEVICE_MAX_MEM_ALLOC_SIZE
        cl_ulong max_mem_alloc_size;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
        printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte\n", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

        // CL_DEVICE_GLOBAL_MEM_SIZE
        cl_ulong mem_size;
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
        printf("CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(mem_size / (1024 * 1024)));

        // CL_DEVICE_ERROR_CORRECTION_SUPPORT
        cl_bool error_correction_support;
        clGetDeviceInfo(devices[i], CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
        printf("CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", error_correction_support == CL_TRUE ? "yes" : "no");

        // CL_DEVICE_LOCAL_MEM_TYPE
        cl_device_local_mem_type local_mem_type;
        clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
        printf("CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", local_mem_type == 1 ? "local" : "global");

        // CL_DEVICE_LOCAL_MEM_SIZE
        clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
        printf("CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte\n", (unsigned int)(mem_size / 1024));

        // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
        printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte\n", (unsigned int)(mem_size / 1024));

        // CL_DEVICE_QUEUE_PROPERTIES
        cl_command_queue_properties queue_properties;
        clGetDeviceInfo(devices[i], CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
        if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
            printf("CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
        if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
            printf("CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_PROFILING_ENABLE");

        // CL_DEVICE_IMAGE_SUPPORT
        cl_bool image_support;
        clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
        printf("CL_DEVICE_IMAGE_SUPPORT:\t\t%u\n", image_support);

        // CL_DEVICE_MAX_READ_IMAGE_ARGS
        cl_uint max_read_image_args;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
        printf("CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u\n", max_read_image_args);

        // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
        cl_uint max_write_image_args;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
        printf("CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u\n", max_write_image_args);

        // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
        size_t szMaxDims[5];
        printf("\nCL_DEVICE_IMAGE <dim>");
        clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
        printf("\t\t\t2D_MAX_WIDTH\t %u\n", szMaxDims[0]);
        clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
        printf("\t\t\t\t\t2D_MAX_HEIGHT\t %u\n", szMaxDims[1]);
        clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
        printf("\t\t\t\t\t3D_MAX_WIDTH\t %u\n", szMaxDims[2]);
        clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
        printf("\t\t\t\t\t3D_MAX_HEIGHT\t %u\n", szMaxDims[3]);
        clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
        printf("\t\t\t\t\t3D_MAX_DEPTH\t %u\n", szMaxDims[4]);

        // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
        printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t");
        cl_uint vec_width [6];
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vec_width[4], NULL);
        clGetDeviceInfo(devices[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vec_width[5], NULL);
        printf("CHAR %u, SHORT %u, INT %u, FLOAT %u, DOUBLE %u\n\n\n",
               vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4]);
    }
}



int main (int argc, char* argv[]) {

    if (argc != 7) {
        std::cerr << "Invalid parameters\n";
        exit(1);
    }

    int outRows, outCols;
    int rows = std::stoi(argv[1]);
    int cols = std::stoi(argv[2]);
    int sheets = std::stoi(argv[3]);
    std::string fileName = argv[4];
    int projectionType = std::stoi(argv[5]);
    std::string outFileName = argv[6];

    // Determine projection size
    int projSize;
    if (projectionType == 1 || projectionType == 2) {
        projSize = rows * cols;
        outRows = rows;
        outCols = cols;
    }
    else if (projectionType == 3 || projectionType == 4) {
        projSize = cols * sheets;
        outRows = rows;
        outCols = sheets;
    }
    else if (projectionType == 5 || projectionType == 6) {
        projSize = rows * sheets;
        outRows = sheets;
        outCols = cols;
    }

    // Read file
    auto fileSize = rows * cols * sheets;
    unsigned char* data;
    std::ifstream file;
    file.open(fileName);

    if (file.is_open()) {
        data = new unsigned char[fileSize];
        std::cout << "Filed opened\n";
        file.read(reinterpret_cast<char*>(data), fileSize);
        file.close();

    } else {
        std::cerr << "File did not open\n";
        exit(1);
    }

    // Create max, working sum, and sum results arrays
    auto maxImg = new unsigned char[projSize];
    auto sumImg = new unsigned char[projSize];
    auto workSum = new float[projSize];

    // Get platform info
    cl_uint numPlatforms = 0;
    cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    auto platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    checkStatus("clGetPlatformIDs", status, true, DEBUG);

    print_platforms(platforms, numPlatforms);

    // Get number of devices for platform 0
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platforms[0], devType, 0, nullptr, &numDevices);
    checkStatus("clGetDeviceIDs-0", status, true, DEBUG);

    // Create device array
    auto devices = new cl_device_id[numDevices];

    // Initialize devices for platform 0
    status = clGetDeviceIDs(platforms[0], devType, numDevices, devices, nullptr);
    checkStatus("clGetDeviceIDs-0", status, true, DEBUG);

    print_devices(devices, numDevices);

    int device;
    while (true) {
        std::cout << "Pick device: ";
        std::cin >> device;
        if (0 <= device && device < numDevices)
            break;
    }

    // Create context
    cl_context context = clCreateContext(nullptr, numDevices, devices, nullptr, nullptr, &status);
    checkStatus("clCreateContext", status, true, DEBUG);

    // Create command queue for specified device
    cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[device], 0, &status);
    checkStatus("clCreateCommandQueue", status, true, DEBUG);

    // Create buffer to send image data to kernel
    auto buffSize = fileSize * sizeof(unsigned char);
    auto imgBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, buffSize, nullptr, &status);
    checkStatus("clCreateBuffer-imgBuffer", status, true, DEBUG);

    // Write image data to buffer
    status = clEnqueueWriteBuffer(cmdQueue, imgBuffer, CL_FALSE, 0, buffSize, data, 0, nullptr, nullptr);
    checkStatus("clEnqueueWriteBuffer-imgBuffer", status, true, DEBUG);

    // Create buffer to put max results in
    auto maxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffSize, nullptr, &status);
    checkStatus("clCreateBuffer-maxBuffer", status, true, DEBUG);

    // Create sum buffer
    auto sumBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffSize, nullptr, &status);
    checkStatus("clCreateBuffer-sumBuffer", status, true, DEBUG);

    // Create buffer for working sum buffer
    auto workBuffSize = projSize * sizeof(float);
    auto workingBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, workBuffSize, nullptr, &status);
    checkStatus("clCreateBuffer-workingBuffer", status, true, DEBUG);

    // Get kernel 1
    const char* programSource[] = { readSource("MaxKernel.cl") };
    cl_program program = clCreateProgramWithSource(context, 1, programSource, nullptr, &status);
    checkStatus("clCreateProgramWithSource", status, true, DEBUG);

    status = clBuildProgram(program, numDevices, devices,
                            nullptr, nullptr, nullptr);
    if (status != 0)
        showProgramBuildLog(program, devices[device]);
    checkStatus("clBuildProgram", status, true, DEBUG);

    // Create Kernel 1
    cl_kernel kernel = clCreateKernel(program, "MaxKernel", &status);
    checkStatus("clCreateKernel", status, true, DEBUG);

    // Set Kernel 1 args
    status = clSetKernelArg(kernel, 0, sizeof(int), &sheets);
    checkStatus("clSetKernelArg-0", status, true, DEBUG);
    status = clSetKernelArg(kernel, 1, sizeof(int), &cols);
    checkStatus("clSetKernelArg-1", status, true, DEBUG);
    status = clSetKernelArg(kernel, 2, sizeof(int), &rows);
    checkStatus("clSetKernelArg-2", status, true, DEBUG);
    status = clSetKernelArg(kernel, 3, sizeof(int), &projectionType);
    checkStatus("clSetKernelArg-3", status, true, DEBUG);
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &imgBuffer);
    checkStatus("clSetKernelArg-4", status, true, DEBUG);
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &maxBuffer);
    checkStatus("clSetKernelArg-5", status, true, DEBUG);
    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &workingBuffer);
    checkStatus("clSetKernelArg-6", status, true, DEBUG);

    // Set processing size
    size_t global_work_size[] = { 256, 256 };
    size_t* global_work_offset = nullptr;
    // size_t local_work_size[] = { 32, 32 };
    size_t* local_work_size = nullptr;

    // Run Kernel 1
    status = clEnqueueNDRangeKernel(
            cmdQueue,
            kernel,
            2,
            global_work_offset,
            global_work_size,
            local_work_size,
            0,
            nullptr,
            nullptr
        );
    checkStatus("clEnqueueNDRangeKernel-1", status, true, DEBUG);

    /*
    Read data back
    TODO: this is where I seg fault. I've doubled checked sizes and parameters agaisnt
    provided examples as well as tried to change the 5th parameter to 'progSize' which
    did not cause a seg fault but the max image prints nothing and there is no working
    sum max
    */
    clEnqueueReadBuffer(cmdQueue, maxBuffer, CL_FALSE, 0, buffSize, maxImg, 0, nullptr, nullptr);
    clEnqueueReadBuffer(cmdQueue, workingBuffer, CL_FALSE, 0, workBuffSize, workSum, 0, nullptr, nullptr);

    // Block until finished
    clFinish(cmdQueue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    // Write max image
    auto ImgWriter = ImageWriter::create(outFileName + "Max.jpeg", outCols, outRows);
    ImgWriter->writeImage(maxImg);
    delete ImgWriter;

    // Find max of working sum
    float max = 0.00;
    for (int x = 0; x < outCols; x++) {
        for (int y = 0; y < outRows; y++) {
            auto idx = x * cols + y;
            auto val = workSum[idx];
            if (val > max) {
                max = val;
            }
        }
    }
    std::cout << "Max sum value: " << max << std::endl;

    // Get kernel 2
    const char* sumSource[] = { readSource("SumKernel.cl") };
    program = clCreateProgramWithSource(context, 1, sumSource, nullptr, &status);
    checkStatus("clCreateProgramWithSource", status, true, DEBUG);

    status = clBuildProgram(program, numDevices, devices,
                            nullptr, nullptr, nullptr);
    if (status != 0)
        showProgramBuildLog(program, devices[device]);
    checkStatus("clBuildProgram", status, true, DEBUG);

    // Create Sum kernel
    cl_kernel SumKernel = clCreateKernel(program, "SumKernel", &status);
    checkStatus("clCreateKernel", status, true, DEBUG);

    // Set sum kernel args
    status = clSetKernelArg(kernel, 0, sizeof(int), &cols);
    checkStatus("clSetKernelArg-0", status, true, DEBUG);
    status = clSetKernelArg(kernel, 1, sizeof(float), &max);
    checkStatus("clSetKernelArg-1", status, true, DEBUG);
    status = clSetKernelArg(kernel, 2, workBuffSize, &workingBuffer);
    checkStatus("clSetKernelArg-2", status, true, DEBUG);
    status = clSetKernelArg(kernel, 3, buffSize, &sumBuffer);
    checkStatus("clSetKernelArg-3", status, true, DEBUG);

    // Run sum kernel
    status = clEnqueueNDRangeKernel(
            cmdQueue,
            SumKernel,
            2,
            global_work_offset,
            global_work_size,
            local_work_size,
            0,
            nullptr,
            nullptr
        );
    checkStatus("clEnqueueNDRangeKernel-2", status, true, DEBUG);

    // Read back sum image data
    clEnqueueReadBuffer(cmdQueue, sumBuffer, CL_TRUE, 0, buffSize, sumImg, 0, nullptr, nullptr);
    clFinish(cmdQueue);

    // Write out image
    ImgWriter = ImageWriter::create(outFileName + "Sum.jpeg", outCols, outRows);
    ImgWriter->writeImage(sumImg);
    delete ImgWriter;

    // Free OpenCL resources
    clReleaseMemObject(imgBuffer);
    clReleaseMemObject(maxBuffer);
    clReleaseMemObject(sumBuffer);
    clReleaseMemObject(workingBuffer);
    clReleaseKernel(SumKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);

    // Clean up
    delete[] data;
    delete[] maxImg;
    delete[] sumImg;
    delete[] platforms;
    delete[] devices;
    delete[] programSource;
    delete[] sumSource;
}
