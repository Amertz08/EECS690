#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>

#include "ImageReader.h"

void mapToArray(cryph::Packed3DArray<unsigned char>* img, char* data)
{
    int prevSize = 0;
    for (int r = 0; r < img->getDim1(); r++) {
        for (int c = 0; c < img->getDim2(); c++) {
            for (int rgb = 0; rgb < img->getDim3(); rgb++) {
                data[prevSize + r + c + rgb] = img->getDataElement(r, c, rgb);
            }
        }
    }

}

void print_imgs(std::vector<std::string>* l)
{
    std::cout << "Image list\n";
    for (auto const& entry : *l) {
        std::cout << entry << std::endl;
    }
}

void print(std::string msg)
{
    std::cout << msg << std::endl;
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
    // std::cout << "imgCount: " << imgCount << std::endl;
    int processingRanks = rankCount - 1;
    if (imgCount != processingRanks) {
        std::cerr << "Rank count and image count mismatch. processingRanks = "
                  << processingRanks << " imgCount = " << imgCount << std::endl;
        for (int i = 0; i < argc; i++) {
            std::cout << argv[i] << " ";
        }
        std::cout << std::endl;
        exit(1);
    }

    int msgTag = 1;
    if (rank == 0) {

        // Get image data and dimension data from image
        auto dims = new int*[imgCount];
        auto images = new cryph::Packed3DArray<unsigned char>*[imgCount];
        for (int i = 0; i < imgCount; i++) {
            auto file = argv[i + 1];
            std::cout << "Reading file: " << file << std::endl;
            auto ir = ImageReader::create(file);
            if (ir == nullptr) {
                std::cerr << "Could not open image file " << file << std::endl;
                exit(1);
            }
            auto pa = ir->getInternalPacked3DArrayImage();

            images[i] = pa;
            dims[i] = new int[3];
            dims[i][0] = images[i]->getDim1();
            dims[i][1] = images[i]->getDim2();
            dims[i][2] = images[i]->getDim3();
            delete ir;
        }

        // Requests to rank > 0
        auto reqs = new MPI_Request[imgCount - 1];

        std::cout << "rank 0: sending dimension data...\n";
        // Send dimension data to ranks > 0
        for (int i = 1; i < imgCount; i++) {
            std::cout << "sending dimensions to rank: " << i << std::endl;
            MPI_Send(dims[i], 3, MPI_INT, i, msgTag, MPI_COMM_WORLD);
        }

        // Send data
        for (int i = 0; i < imgCount; i++) {
            auto img = images[i];
            if (i != 0) {
                auto data = img->getData();
                auto size = img->getTotalNumberElements();
                MPI_Isend(data, size, MPI_UNSIGNED_CHAR, i, msgTag, MPI_COMM_WORLD, &reqs[i]);
            }
        }

        // Do rank 0 calculations

    } else {
        int recDims[3];
        MPI_Status status;
        std::cout << "rank: " << rank << " waiting on dimensions\n";
        MPI_Recv(&recDims, 3, MPI_INT, 0, msgTag, MPI_COMM_WORLD, &status);
        int size = recDims[0] * recDims[1] * recDims[2];
        std::cout << "rank: " << rank << " image size received: " << size << std::endl;

        // Read image data
        auto dataBuffer = new unsigned char[size];
        std::cout << "rank: " << rank << " waiting on data\n";
        MPI_Recv(dataBuffer, size, MPI_UNSIGNED_CHAR, 0, msgTag, MPI_COMM_WORLD, &status);
        std::cout << "rank: " << rank << " read image\n";
        delete[] dataBuffer;
    }

    MPI_Finalize();
    return 0;

}
