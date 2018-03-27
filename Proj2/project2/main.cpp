#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <mpi.h>

#include "ImageReader.h"

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
        auto dims = new int*[imgCount - 1];
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

            images[i + 1] = pa;
            dims[i] = new int[3];
            dims[i][0] = images[i + 1]->getDim1();
            dims[i][1] = images[i + 1]->getDim2();
            dims[i][2] = images[i + 1]->getDim3();
            delete ir;
        }

        // Requests to rank > 0
        auto reqs = new MPI_Request[imgCount - 1];

        std::cout << "rank 0: sending dimension data...\n";
        // Send dimension data to ranks > 0
        for (int i = 0; i < imgCount - 1; i++) {
            std::cout << "sending dimensions to rank: " << i + 1 << std::endl;
            MPI_Send(dims[i], 3, MPI_INT, i + 1, msgTag, MPI_COMM_WORLD);
        }

        // Send data
        for (int i = 0; i < imgCount - 1; i++) {
            auto img = images[i + 1];
            auto data = img->getData();
            auto size = img->getTotalNumberElements();
            MPI_Isend(data, size, MPI_UNSIGNED_CHAR, i + 1, msgTag, MPI_COMM_WORLD, &reqs[i]);
        }

        /*
         * Do rank 0 calculations
         */
        auto weights = new float**[3];

        weights[0] = CalculateHistogram(images[0]);


        // TODO: delete data structures
//        for (int i = 0; i < imgCount - 1; i++) {
//            delete[] dims[i];
//        }
//        delete[] dims;
//        delete[] images;
//        delete[] weights;
//        delete[] reqs;
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
        std::cout << "rank: " << rank << " data received\n";

        // Do histogram calculations
        cryph::Packed3DArray<unsigned char> array(recDims[0], recDims[1], recDims[2], dataBuffer);
        delete[] dataBuffer;

        auto hist = CalculateHistogram(&array);

        // TODO calculate histogram
    }

    MPI_Finalize();
    return 0;

}
