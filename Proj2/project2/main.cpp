#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>

#include "ImageReader.h"

#define BUFFER_MAX 100

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
    // std::cout << "imgCount: " << imgCount << std::endl;
    int processingRanks = rankCount - 1;
    if (imgCount != processingRanks) {
        std::cerr << "Rank count and image count missmatch. processingRanks = "
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
        std::vector<const unsigned char*> imgData;
        std::vector<int*> dims;
        for (int i = 1; i < imgCount; i++) {
            auto ir = ImageReader::create(argv[i]);
            if (ir == nullptr) {
                std::cerr << "Could not open image file " << argv[i] << std::endl;
                exit(1);
            }
            cryph::Packed3DArray<unsigned char>* pa = ir->getInternalPacked3DArrayImage();
            imgData.push_back(pa->getData());
            int* d = new int[3];
            d[0] = pa->getDim1();
            d[1] = pa->getDim2();
            d[2] = pa->getDim3();
            dims.push_back(d);
            delete ir;
        }


    } else {

    }

    MPI_Finalize();
    return 0;

}
