#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "ImageReader.h"

float* FlattenHist(float **h)
{
    auto data = new float[256 * 3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 256; j++) {
            data[j + i * 256] = h[i][j];
        }
    }
}

float** RebuildHist(float* h){
    auto colors = new float*[3];
    for(int i = 0; i < 3; i++) {
        colors[i] = new float[256];
        for (int j = 0; j < 256; j++) {
            colors[i][j] = h[j + i * 256];
        }
    }
    return colors;
}

float ArrayDiff(float* a, float* b)
{
    float diff = 0.0;
    for (int i = 0; i < 256; i++) {
        diff += abs(a[i] - b[i]);
    }
    return diff;
}

float* ColorScore(float** a, float** b)
{
    auto s = new float[3]();
    for (int i = 0; i < 3; i++) {
        s[i] = ArrayDiff(a[i], b[i]);
    }
    return s;
}

float Score(float** a, float** b)
{
    auto scores = ColorScore(a, b);
    float s = 0.0;
    for (int i = 0; i < 3; i++) {
        s += scores[i];
    }
    delete[] scores;
    return s;
}

int** ColorCount(cryph::Packed3DArray<unsigned char>* pa)
{
    auto colorCount = new int*[3];
    for (int i = 0; i < 3; i++) {
        colorCount[i] = new int[256]();
    }
    // Tally color count
    for (int r = 0; r < pa->getDim1(); r++) {
        for (int c = 0; c < pa->getDim2(); c++) {
            for (int rgb = 0; rgb < pa->getDim3(); rgb++) {
                // calculate
                auto el = (int)pa->getDataElement(r, c, rgb);
                colorCount[rgb][el] += 1;
            }
        }
    }
    return colorCount;
}

float** CalculateHistogram(cryph::Packed3DArray<unsigned char>* pa)
{

    auto colorCount = ColorCount(pa);
    auto proportions = new float*[3];

    auto denominator = (float)pa->getTotalNumberElements() / 3;

    for (int i = 0; i < 3; i++) {
        proportions[i] = new float[256]();
        for (int j = 0; j < 255; j++) {
            proportions[i][j] = (float)colorCount[i][j] / denominator;
        }
    }

    for (int i = 0; i < 3; i++) {
        delete[] colorCount[i];
    }
    delete[] colorCount;
    return proportions;
}

void print_imgs(std::vector<std::string>* l)
{
    std::cout << "Image list\n";
    for (auto const& entry : *l) {
        std::cout << entry << std::endl;
    }
}


int main(int argc, char* argv[]) {
    // Setup MPI
    MPI_Init(&argc, &argv);
    int rank, rankCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    // Check CLI inputs
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "image.jpeg image2.jpeg ...\n";
        exit(1);
    }

    // Calculate and check image count
    int imgCount = argc - 1;
    if (imgCount != rankCount) {
        if (rank == 0) {
            std::cerr << "Rank count and image count mismatch. ranks = "
                      << rankCount << " imgCount = " << imgCount << std::endl;
            for (int i = 0; i < argc; i++) {
                std::cout << argv[i] << " ";
            }
            std::cout << std::endl;
        }
        exit(1);
    }
    std::cout << "Image count: " << imgCount << " Ranks: " << rankCount << std::endl;

    srand (time(NULL));
    int msgTag = 1;
    if (rank == 0) {

        cryph::Packed3DArray<unsigned char>* localImage;
        for (int i = 0; i < imgCount; i++) {
            auto file = argv[i + 1];
            std::cout << "Reading file: " << file << std::endl;
            auto ir = ImageReader::create(file);
            if (ir == nullptr) {
                std::cerr << "Could not open image file " << file << std::endl;
                exit(1);
            }
            auto pa = ir->getInternalPacked3DArrayImage();

            if (i == 0) {
                std::cout << "assigning rank 0 image\n";
                localImage = pa;
            } else {
                auto dims = new int[3]();
                auto a = pa->getDim1();
                auto b = pa->getDim2();
                auto c = pa->getDim3();
                auto size = a * b * c;
                dims[0] = a;
                dims[1] = b;
                dims[2] = c;
                auto data = pa->getData();
                std::cout << "Sending rank: " << i << " dimensions\n";
                MPI_Send(dims, 3, MPI_INT, i, msgTag, MPI_COMM_WORLD);
                std::cout << "Sending rank: " << i << " data of size: " << size << std::endl;
                MPI_Send(data, size, MPI_UNSIGNED_CHAR, i, msgTag, MPI_COMM_WORLD);
                delete[] dims;
                delete[] data;
            }
        }

        /*
         * Do rank 0 calculations
         */
        float** weights[3];
        weights[0] = CalculateHistogram(localImage);
        int flatSize = 256 * 3;
        auto flatData = new float*[3];
        flatData[0] = FlattenHist(weights[0]);

        // Get data from ranks > 0
        MPI_Status status;
        for (int i = 0; i < imgCount; i++) {
            if (i != rank) {
                flatData[i] = new float[flatSize];
                MPI_Recv(flatData[i], flatSize, MPI_FLOAT, i, msgTag, MPI_COMM_WORLD, &status);
            }
        }

        // Send data back out to ranks
        MPI_Request reqs[imgCount - 1];
        for (int i = 0; i < imgCount; i++) {
            if (i != rank) {
                std::cout << "Sending proportion data to rank: " << i << std::endl;
                MPI_Isend(flatData[i], flatSize, MPI_FLOAT, i, msgTag, MPI_COMM_WORLD, &reqs[i - 1]);
            }
        }


        // TODO: delete objects
//        delete[] weights;
        delete localImage;
    } else {
        int recDims[3];
        MPI_Status status;
        std::cout << "rank: " << rank << " waiting on dimensions\n";
        MPI_Recv(&recDims, 3, MPI_INT, 0, msgTag, MPI_COMM_WORLD, &status);
        int totalSize = recDims[0] * recDims[1] * recDims[2];
        std::cout << "rank: " << rank << " image size received: " << totalSize << std::endl;

        // Read image data
        auto dataBuffer = new unsigned char[totalSize];
        std::cout << "rank: " << rank << " waiting on image data\n";
        MPI_Recv(dataBuffer, totalSize, MPI_UNSIGNED_CHAR, 0, msgTag, MPI_COMM_WORLD, &status);
        std::cout << "rank: " << rank << " image data received\n";

        // Do histogram calculations
        cryph::Packed3DArray<unsigned char> array(recDims[0], recDims[1], recDims[2], dataBuffer);
        delete[] dataBuffer;

        auto hist = CalculateHistogram(&array);
        auto flatHist = FlattenHist(hist);

        // Send histogram back to rank 0
        int flatSize = 256 * 3;
        auto flatData = new float*[3];
        flatData[rank] = flatHist;
        MPI_Send(flatHist, flatSize, MPI_FLOAT, 0, msgTag, MPI_COMM_WORLD);

        // Get all other ranks data
        for (int i = 0; i < imgCount; i++) {
            if (i != rank) {
                std::cout << "rank: " << rank << " waiting on proportion data\n";
                flatData[i] = new float[flatSize];
                MPI_Recv(flatData[i], flatSize, MPI_FLOAT, 0, msgTag, MPI_COMM_WORLD, &status);
            }
        }

        // TODO: delete objects
        delete hist;
    }

    MPI_Finalize();
    return 0;

}
