#include <iostream>
#include <string>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif


int main (int argc, char* argv[]) {
    cl_uint numPlatforms = 0;
    cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);

    auto platforms = new cl_platform_id[numPlatforms];

    status = clGetPlatformIDs(numPlatforms, platforms, nullptr);


    delete[] platforms;
}