#ifndef EECS690_HELPERS_HPP
#define EECS690_HELPERS_HPP

/**
 * These are all functions defined by Prof. Miller
 */


struct NameTable
{
    std::string name;
    int value;
};

void reportPlatformInformation(const cl_platform_id& platformIn)
{
    NameTable what[] = {
            { "CL_PLATFORM_PROFILE:    ", CL_PLATFORM_PROFILE },
            { "CL_PLATFORM_VERSION:    ", CL_PLATFORM_VERSION },
            { "CL_PLATFORM_NAME:       ", CL_PLATFORM_NAME },
            { "CL_PLATFORM_VENDOR:     ", CL_PLATFORM_VENDOR },
            { "CL_PLATFORM_EXTENSIONS: ", CL_PLATFORM_EXTENSIONS },
            { "", 0 }
    };
    size_t size;
    char* buf = nullptr;
    int bufLength = 0;
    std::cout << "===============================================\n";
    std::cout << "========== PLATFORM INFORMATION ===============\n";
    std::cout << "===============================================\n";
    for (int i=0 ; what[i].value != 0 ; i++)
    {
        clGetPlatformInfo(platformIn, what[i].value, 0, nullptr, &size);
        if (size > bufLength)
        {
            if (buf != nullptr)
                delete [] buf;
            buf = new char[size];
            bufLength = size;
        }
        clGetPlatformInfo(platformIn, what[i].value, bufLength, buf, &size);
        std::cout << what[i].name << buf << '\n';
    }
    std::cout << "================= END =========================\n\n";
}

// A common utility extracted from the source code associated with
// the book "Heterogeneous Computing with OpenCL".

// This function reads in a text file and stores it as a char pointer
const char* readSource(const char* kernelPath) {

    FILE *fp;
    char *source;
    long int size;

    printf("Program file is: %s\n", kernelPath);

    fp = fopen(kernelPath, "rb");
    if(!fp) {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    int status = fseek(fp, 0, SEEK_END);
    if(status != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if(size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    rewind(fp);

    source = (char *)malloc(size + 1);

    int i;
    for (i = 0; i < size+1; i++) {
        source[i]='\0';
    }

    if(source == NULL) {
        printf("Error allocating space for the kernel source\n");
        exit(-1);
    }

    fread(source, 1, size, fp);
    source[size] = '\0';

    //printf("Returning file:\n%s\n", source);

    return source;
}

void checkStatus(std::string where, cl_int status, bool abortOnError, bool debug)
{
    if (debug || (status != 0))
        std::cout << "Step " << where << ", status = " << status << '\n';
    if ((status != 0) && abortOnError)
        exit(1);
}

void showProgramBuildLog(cl_program pgm, cl_device_id dev)
{
    size_t size;
    clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
    char* log = new char[size+1];
    clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, size+1, log, nullptr);
    std::cout << "LOG:\n" << log << "\n\n";
    delete [] log;
}

#endif //EECS690_HELPERS_HPP
