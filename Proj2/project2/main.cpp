#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "ImageReader.h"

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
        diff += fabs(a[i] - b[i]);
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
//            scores[i] = 5;
        }
    }
    return scores;
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

    auto denominator = (float)(pa->getTotalNumberElements()) / 3;

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

float** UnFlatten2D(float* data, int r, int c)
{
    auto d = new float*[r];
    for (int i = 0; i < r; i++) {
        d[i] = new float[c];
        for (int j = 0; j < c; j++) {
            d[i][j] = data[j + (i * c)];
        }
    }
    return d;
}

/**
 * axb -> [a*b]
 * @param data
 * @param r : row
 * @param c : column
 * @return
 */
float* Flatten2D(float** data, int r, int c)
{
    auto d = new float[r * c];
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            d[j + (i * c)] = data[i][j];
        }
    }
    return d;
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

    srand (time(NULL));
    int msgTag = 1;
    int flatSize = 256  * 3;
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
        auto hist = CalculateHistogram(localImage); // 3 x 256
        auto flatHist = Flatten2D(hist, 3, 256); // 1 x 768


        // Get data from ranks > 0
        MPI_Status status;
        std::cout << "Gathering proportion data from other processes\n";
        auto recFlat = new float[(imgCount - 1) * flatSize];
        MPI_Gather(nullptr, 0, MPI_FLOAT, recFlat, flatSize, MPI_FLOAT, rank, MPI_COMM_WORLD);


        // store data
        auto flatData = new float*[imgCount]; // imgCount x 768
        for (int i = 0; i < imgCount; i++) {
            if (i != rank)
                flatData[i] = &recFlat[i - 1]; // 1 x 768
            else
                flatData[i] = flatHist;
        }



        auto flat = Flatten2D(flatData, imgCount, flatSize); // imgCount * 768
        std::cout << "Sending data back out to ranks\n";
        MPI_Allgather(flat, imgCount * flatSize, MPI_FLOAT, nullptr, 0, MPI_FLOAT, MPI_COMM_WORLD);
        std::cout << "Data sent\n";


        // Rebuild data
        auto weights = new float**[imgCount]; // Nx3x256
        for (int i = 0; i < imgCount; i++) {
            if (i != rank) {
                std::cout << "rank 0 rebuilding: " << i << std::endl;
                weights[i] = UnFlatten2D(flat, imgCount, 768); // imgCount x 768
            } else {
                weights[i] = hist;
            }
        }

        // Get scores from other ranks
        auto scores = GetScores(weights, imgCount, rank); // 1 x imgCount
        for (int i = 0; i < imgCount; i++) {
            std::cout << "rank 0 score: " << scores[i] << std::endl;
        }
        auto allScores = new float[imgCount * imgCount];
        // receiving imgCount * imgCount
        MPI_Gather(nullptr, 0, MPI_FLOAT, allScores, imgCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
        auto blah = UnFlatten2D(allScores, imgCount, imgCount);

        for (int i = 0; i < imgCount * imgCount; i++) {
            std::cout << allScores[i] << std::endl;
        }

        for (int i = 0; i < imgCount; i++) {
            PrintScores(blah[i], imgCount, i);
        }
        std::cout << "rank 0 made it to the end\n";
    } else {
        MPI_Request req;

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

        auto hist = CalculateHistogram(&array); // 3 x 256
        auto flatHist = Flatten2D(hist, 3, 256); // 1 x 768

        // Send histograms back to rank 0
        auto flatData = new float*[imgCount]; // imgCount x 768
        flatData[rank] = flatHist;
        std::cout << "rank: " << rank << " sending histogram back to rank 0\n";
        MPI_Gather(flatHist, flatSize, MPI_FLOAT, nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
        std::cout << "rank: " << rank << " histogram sent\n";

        // Get all other ranks data
        auto flat = new float[flatSize * imgCount]; // imgCount * 768
        std::cout << "rank: " << rank << " waiting on other rank data\n";
        MPI_Allgather(nullptr, 0, MPI_FLOAT, flat, imgCount * flatSize, MPI_FLOAT, MPI_COMM_WORLD);
        std::cout << "rank: " << rank << " received data on other ranks\n";

        auto ss = UnFlatten2D(flat, imgCount, flatSize); // imgCount x 768


        // Unflatten data
        auto weights = new float**[imgCount]; // imgCount x 3 x 256
        for (int i = 0; i < imgCount; i++) {
            if (i != rank)
                weights[i] = UnFlatten2D(ss[i], imgCount, 768); // imgCount x 768 -> 3 x 256
            else
                weights[i] = hist;
        }

        // Calculate scores
        auto scores = GetScores(weights, imgCount, rank);
        std::cout << "rank: " << rank << " calculated scores\n";

        for (int i = 0; i < imgCount; i++) {
            std::cout << "rank: " << rank << " scores: " << scores[i] << std::endl;
        }

        MPI_Gather(scores, imgCount, MPI_FLOAT, nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD); // sending 1 x imgCount

        // TODO: delete objects
        std::cout << "rank: " << rank << " made it to the end\n";
    }

    MPI_Finalize();
    return 0;

}
