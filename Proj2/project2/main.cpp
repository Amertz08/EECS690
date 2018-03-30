#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "ImageReader.h"

/**
 * Flattens 3x256 into 768 float array
 * @param h : float[3][256]
 * @return : float[768]
 */
float* FlattenHist(float **h)
{
    auto data = new float[256 * 3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 256; j++) {
            data[j + i * 256] = h[i][j];
        }
    }
}

/**
 * Takes 768 length histogram and rebuilds to 3x256
 * @param h : float[768]
 * @return : float[3][256]
 */
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

/**
 * Sum of the magnitude difference in two normalized histograms of color
 * @param a : normalized histogram of a specific color
 * @param b : normalized histogram of a specific color
 * @return sigma(|a[i] - b[i]|)
 */
float ArrayDiff(float* a, float* b)
{
    float diff = 0;
    for (int i = 0; i < 256; i++) {
        diff += abs(a[i] - b[i]);
    }
    return diff;
}

/**
 * Calculates the total magnitude difference for a specific color in the images
 * @param a : R, G, or B normalized frequency distribution for an image
 * @param b : R, G, or B normalized frequency distribution for an image
 * @return the sum of the absolute value of the difference of each entry
 */
float* ColorScore(float** a, float** b)
{
    auto s = new float[3];
    for (int i = 0; i < 3; i++) {
        s[i] = ArrayDiff(a[i], b[i]);
    }
    return s;
}

/**
 * Calculates the simularity score for the given image proportion data
 * @param a : 3x256 normalized histogram for RGB colors
 * @param b : 3x256 normalized histogram for RGB colors
 * @return : Sum of the total difference between each of the 3 colors.
 */
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

/**
 * Calculates the scores for the given rank
 * @param weights
 * @param imgCount
 * @param rank
 * @return
 */
float* GetScores(float*** weights, int imgCount, int rank)
{
    std::cout << "rank: " << rank << " calculating scores\n";
    auto scores = new float[imgCount]();
    for (int i = 0; i < imgCount; i++) {
        if (i != rank) {
            scores[i] = Score(weights[rank], weights[i]);
        }
    }
}

/**
 * Calculates the frequency histograms for RGB color values in the image
 * @param pa : Packed3DArray object
 * @return : 3 x 256 int array of RGB color frequency
 */
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

/**
 * Calculates the normalized histogram for the image
 * @param pa : Packed3DArray object
 * @return : 3 x 256 float array with normalized frequency distributions of RGB colors
 */
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

void PrintScores(float* scores, int imageCount, int rank)
{
    for (int i = 0; i < imageCount; i++) {
        std::cout << "rank: " << rank << " scores[" << i << "]: " << scores[i] << std::endl;
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
//    std::cout << "Image count: " << imgCount << " Ranks: " << rankCount << std::endl;

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
                std::cout << "rank: 0 getting proportion data from " << i << std::endl;
                MPI_Recv(flatData[i], flatSize, MPI_FLOAT, i, msgTag, MPI_COMM_WORLD, &status);
            }
        }


        // Rebuild data
        for (int i = 0; i < imgCount; i++) {
            if (i != rank)
                weights[i] = RebuildHist(flatData[i]);
        }

        // Send data back out to ranks
        for (int r = 0; r < imgCount; r++) { // For each rank
            std::cout << "Sending proportion data to rank: " << r << std::endl;
            for (int i = 0; i < imgCount; i++) { // Send each image except its own
                if (i != r) { // When target rank isn't data rank
                    std::cout << "sending i: " << i << " to: " << r << std::endl;
                    MPI_Send(flatData[i], flatSize, MPI_FLOAT, r, msgTag, MPI_COMM_WORLD); // (imgCount - 1) ** 2 sends
                }
            }
        }

        auto scores = GetScores(weights, imgCount, rank);
//        PrintScores(scores, imgCount, rank);



        // TODO: delete objects
//        delete[] weights;
        delete localImage;
    } else {
        // Get dimensions
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
        std::cout << "rank: " << rank << " sending histogram back to rank 0\n";
        MPI_Send(flatHist, flatSize, MPI_FLOAT, 0, msgTag, MPI_COMM_WORLD);

        // Get all other ranks data
        for (int i = 0; i < imgCount; i++) {
            if (i != rank) {
                std::cout << "rank: " << rank << " waiting on proportion data for: " << i << std::endl;
                flatData[i] = new float[flatSize];
                MPI_Recv(flatData[i], flatSize, MPI_FLOAT, 0, msgTag, MPI_COMM_WORLD, &status); // (imgCount - 1) ** 2 Rec
            }
        }
//        MPI_Recv(flatData, flatSize, MPI_FLOAT, 0, msgTag, MPI_COMM_WORLD, &status);
        std::cout << "rank: " << rank << " made it out\n";

        // Unflatten data
        float** weights[3];
        for (int i = 0; i < imgCount; i++) {
            if (i != rank)
                weights[i] = RebuildHist(flatData[i]);
            else
                weights[i] = hist;
        }

        // Calculate scores
        auto scores = GetScores(weights, imgCount, rank);

//        for (int i = 0; i < imgCount; i++) {
//            std::cout << "Score: " << scores[i] << std::endl;
//        }


        // TODO: delete objects
//        delete hist;
    }

    MPI_Finalize();
    return 0;

}
