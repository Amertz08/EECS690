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
        int maxImgSize = 0;
        std::vector<cryph::Packed3DArray<unsigned char>*> images;
        for (int i = 1; i < imgCount + 1; i++) {
            auto file = argv[i];
            std::cout << "Reading file: " << file << std::endl;
            auto ir = ImageReader::create(file);
            if (ir == nullptr) {
                std::cerr << "Could not open image file " << file << std::endl;
                exit(1);
            }
            auto pa = ir->getInternalPacked3DArrayImage();
            if (pa->getTotalNumberElements() > maxImgSize)
                maxImgSize = pa->getTotalNumberElements();

            images.push_back(pa);
            delete ir;
        }

        // Map Image data to char array
        auto data = new char[maxImgSize];
        auto reqs = new MPI_Request[imgCount];

        // Send char data
        for (int i = 1; i < imgCount + 1; i++) {
            mapToArray(images.at(i - 1), data);
            std::cout << "Sending data to rank: " << i << std::endl;
            // Send immediate so we can continue calculations
            MPI_Isend(data, maxImgSize, MPI_UNSIGNED_CHAR, i, msgTag, MPI_COMM_WORLD, &reqs[i]);
        }

        // Send int data
        int* dims;
        for (int i = 1; i < imgCount + 1; i++) {
            auto img = images.at(i - 1);
            dims = new int[3];
            dims[0] = img->getDim1();
            dims[1] = img->getDim2();
            dims[2] = img->getDim3();
            MPI_Isend(dims, 3, MPI_INT, i, msgTag, MPI_COMM_WORLD, &reqs[i]);
            delete dims;
        }

    } else {
        MPI_Status status;
        // Wait till we get a message
        MPI_Probe(0, msgTag, MPI_COMM_WORLD, &status);

        // Get size
        int size;
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &size);
        std::cout << "rank: " << rank << " image size recieved: " << size << std::endl;

        // Read image data
        auto dataBuffer = new unsigned char[size];
        MPI_Recv(dataBuffer, size, MPI_UNSIGNED_CHAR, 0, msgTag, MPI_COMM_WORLD, &status);
        std::cout << "rank: " << rank << " read image\n";

        // Wait till we get a message
        MPI_Probe(0, msgTag, MPI_COMM_WORLD, &status);

        // Get size
        MPI_Get_count(&status, MPI_INT, &size);
        std::cout << "rank: " << rank << " dimensions size recieved: " << size << std::endl;

        // Read dimension data
        auto dims = new int[size];
        MPI_Recv(dims, size, MPI_INT, 0, msgTag, MPI_COMM_WORLD, &status);
        std::cout << "rank: " << rank << " read ints\n";

        // Build out array
        auto pa = new cryph::Packed3DArray<unsigned char>(dims[0], dims[1], dims[2], dataBuffer);
        delete dataBuffer;
        delete dims;
    }

    MPI_Finalize();
    return 0;

}
