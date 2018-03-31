#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "ImageReader.h"

#define DEBUG 1
#define COLORS 3
#define RANGE 256

/**
 * Find sum of given array
 * @param arr
 * @param size
 * @return
 */
float SumArray(float* arr, int size)
{
    float s = 0.00;
    for (int i = 0; i < size; i++) {
        s += arr[i];
    }
    return s;
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
#if DEBUG
    auto sumA = SumArray(a, RANGE);
    auto sumB = SumArray(b, RANGE);
    std::cout << "sumA: " << sumA << " sumB: " << sumB << std::endl;
#endif
    for (int i = 0; i < RANGE; i++) {
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
    auto s = new float[COLORS];
    for (int i = 0; i < COLORS; i++) {
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
    for (int i = 0; i < COLORS; i++) {
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
    std::cout << "rank " << rank << ": calculating scores\n";
    auto scores = new float[imgCount]();
    for (int i = 0; i < imgCount; i++) {
        if (i != rank) {
            scores[i] = Score(weights[rank], weights[i]);
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
    auto colorCount = new int*[COLORS];
    for (int i = 0; i < COLORS; i++) {
        colorCount[i] = new int[RANGE]();
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
    auto proportions = new float*[COLORS];

    auto denominator = pa->getTotalNumberElements() / 3.0;

    for (int i = 0; i < COLORS; i++) {
        proportions[i] = new float[RANGE]();
        for (int j = 0; j < RANGE; j++) {
            proportions[i][j] = colorCount[i][j] / denominator;
        }
#if DEBUG
        auto sum = SumArray(proportions[i], RANGE);
        std::cout << "i: " << i << " sum: " << sum << "\n";
#endif
    }

    for (int i = 0; i < COLORS; i++) {
        delete[] colorCount[i];
    }
    delete[] colorCount;
    return proportions;
}

/**
 * Prints score array for given rank
 * @param scores : array of scores
 * @param imgCount : total images
 * @param rank
 */
void PrintScores(float* scores, int imgCount, int rank)
{
    for (int i = 0; i < imgCount; i++) {
        std::cout << "rank " << rank << ": scores[" << i << "]: " << scores[i] << std::endl;
    }
}

/**
 * Takes in [r * c] length array and
 * transforms it into [r][c] array
 * @param data : 1D array
 * @param r : row count
 * @param c : column count
 * @return [r][c] array
 */
float** UnFlatten2D(float* data, int r, int c)
{
    auto d = new float*[r];
    for (int i = 0; i < r; i++) {
        d[i] = new float[c];
        for (int j = 0; j < c; j++) {
            d[i][j] = data[j + (i * r)];
        }
    }
    return d;
}

/**
 * Takes [r][c] 2D array and makes [r * c] array
 * @param data : 2D array
 * @param r : row count
 * @param c : column count
 * @return [r * c] array
 */
float* Flatten2D(float** data, int r, int c)
{
    auto d = new float[r * c];
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            d[j + (i * r)] = data[i][j];
        }
    }
    return d;
}

/**
 * Takes in histogram data from other ranks and merges it with
 * the local ranks histogram data
 * @param f : imgCount * 768 length array of normalized frequencies
 * @param localHist : 3 x 256 length array of normalized frequencies
 * @param imgCount : Total images
 * @param rank : current rank
 * @return imgCount x 3 x 256 array of normalized frequency data
 */
float*** BuildWeights(float* f, float** localHist, int imgCount, int rank)
{
    auto ss = UnFlatten2D(f, imgCount, 768); // imgCount * 768 -> imgCount x 768

    // Rebuild data
    auto weights = new float**[imgCount]; // Nx3x256
    for (int i = 0; i < imgCount; i++) {
        if (i != rank) {
            weights[i] = UnFlatten2D(ss[i], COLORS, 768); // 1 x 768 -> 3 x 256
        } else {
            weights[i] = localHist;
        }
    }
    return weights;
}

/**
 * Prints 1d array on same line
 * @param arr : array to print
 * @param size : size of array
 */
void PrintArray(float* arr, int size)
{
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * Prints 2D array
 * @param arr : array to print
 * @param x : row count
 * @param y : column count
 */
void Print2DArray(float** arr, int x, int y)
{
    for (int i = 0; i < x; i++) {
        std::cout << "i: " << i << std::endl;
        PrintArray(arr[i], y);
    }
}

/**
 * Finds the image most like the given rank
 * @param scores : array of scores for the rank
 * @param imgCount : total images
 * @param rank : current rank
 * @return index of image
 */
int FindMostLike(float* scores, int imgCount, int rank)
{
    int index;
    if (rank == 0) {
        index = 1;
    } else {
        index = 0;
    }
    float min = scores[index];
    
    for (int i = 0; i < imgCount; i++) {
        if (scores[i] < min && i != rank) {
            min = scores[i];
            index = i;
        }
    }
    return index;
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

    int msgTag = 1;
    int flatSize = RANGE  * COLORS;
    if (rank == 0) {

        // Read in each image
        cryph::Packed3DArray<unsigned char>* localImage;
        for (int i = 0; i < imgCount; i++) {
            auto file = argv[i + 1];
            std::cout << "rank 0: Reading file: " << file << std::endl;
            auto ir = ImageReader::create(file);
            if (ir == nullptr) {
                std::cerr << "Could not open image file " << file << std::endl;
                exit(1);
            }
            auto pa = ir->getInternalPacked3DArrayImage();

            // Save to local rank
            if (i == 0) {
                std::cout << "rank 0: assigning rank 0 image\n";
                localImage = pa;
            } else {
                // Send data to other processes
                auto dims = new int[3]();
                auto a = pa->getDim1();
                auto b = pa->getDim2();
                auto c = pa->getDim3();
                auto size = a * b * c;
                dims[0] = a;
                dims[1] = b;
                dims[2] = c;
                auto data = pa->getData();
                std::cout << "rank 0: sending rank: " << i << " dimensions\n";
                MPI_Send(dims, 3, MPI_INT, i, msgTag, MPI_COMM_WORLD);
                std::cout << "rank 0: sending rank: " << i << " data of size: " << size << std::endl;
                MPI_Send(data, size, MPI_UNSIGNED_CHAR, i, msgTag, MPI_COMM_WORLD);
                delete[] dims;
                delete[] data;
            }
        }

        /*
         * Do rank 0 calculations
         */
        auto hist = CalculateHistogram(localImage); // 3 x 256
        auto flatHist = Flatten2D(hist, COLORS, RANGE); // 1 x 768


        // Get data from ranks > 0
        MPI_Status status;
        std::cout << "rank 0: Gathering proportion data from other processes\n";
        auto invProp = new float[imgCount * flatSize];
        MPI_Gather(flatHist, flatSize, MPI_FLOAT, invProp, flatSize, MPI_FLOAT, rank, MPI_COMM_WORLD);

        auto invPropFlat = UnFlatten2D(invProp, imgCount, flatSize); // imgCount * 768 -> imgCount x 768

        // Build structure with all ranks proportional data
        auto flatData = new float*[imgCount]; // imgCount x 768
        for (int i = 0; i < imgCount; i++) {
            if (i != rank)
                flatData[i] = invPropFlat[i]; // 1 x 768
            else
                flatData[i] = flatHist;
        }
        // flatData -> imgCount x 768

        // Send all proportion data back out to ranks
        auto flatProportions = Flatten2D(flatData, imgCount, flatSize); // imgCount * 768
        std::cout << "rank 0: Sending data back out to ranks\n";
        MPI_Bcast(flatProportions, imgCount * flatSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
        std::cout << "rank 0: Data sent\n";

        // Rebuild data
        auto weights = BuildWeights(flatProportions, hist, imgCount, rank);

        // Get scores from other ranks
        auto scores = GetScores(weights, imgCount, rank); // 1 x imgCount

        // Combine all scores from other ranks
        auto allScores = new float[imgCount * imgCount];
        // receiving imgCount * imgCount
        MPI_Gather(scores, imgCount, MPI_FLOAT, allScores, imgCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
        auto ScoresMatrix = UnFlatten2D(allScores, imgCount, imgCount);
        std::cout << "rank 0: Received all scores\n";

        // Print out the scores
        for (int i = 0; i < imgCount; i++) {
            PrintScores(ScoresMatrix[i], imgCount, i);
        }

        // Print out similar image
        for (int i = 0; i < imgCount; i++) {
            auto imgIndex = FindMostLike(ScoresMatrix[i], imgCount, i);
            std::cout << "rank " << i << " image: " << argv[i + 1]
                      << " is most like rank " << imgIndex
                      << " image: " << argv[imgIndex + 1] << std::endl;

        }
        std::cout << "rank 0: Finished\n";
        // TODO: memleaks
    } else {
        MPI_Request req;

        // Get dimensions
        int recDims[3];
        MPI_Status status;
        std::cout << "rank " << rank << ": waiting on dimensions\n";
        MPI_Recv(&recDims, 3, MPI_INT, 0, msgTag, MPI_COMM_WORLD, &status);
        int totalSize = recDims[0] * recDims[1] * recDims[2];
        std::cout << "rank " << rank << ": image size received: " << totalSize << std::endl;

        // Read image data
        auto dataBuffer = new unsigned char[totalSize];
        std::cout << "rank " << rank << ": waiting on image data\n";
        MPI_Recv(dataBuffer, totalSize, MPI_UNSIGNED_CHAR, 0, msgTag, MPI_COMM_WORLD, &status);
        std::cout << "rank " << rank << ": image data received\n";

        cryph::Packed3DArray<unsigned char> array(recDims[0], recDims[1], recDims[2], dataBuffer);
        delete[] dataBuffer;

        // Do histogram calculations
        auto hist = CalculateHistogram(&array); // 3 x 256
        auto flatHist = Flatten2D(hist, COLORS, RANGE); // 1 x 768

        // Send histograms back to rank 0
        std::cout << "rank " << rank << ": sending histogram back to rank 0\n";
        MPI_Gather(flatHist, flatSize, MPI_FLOAT, nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
        std::cout << "rank " << rank << ": histogram sent\n";
        std::cout << "rank " << rank << ": printing data sent to gather\n";

        // Get all other ranks data
        auto flatProportions = new float[flatSize * imgCount]; // imgCount * 768
        std::cout << "rank " << rank << ": waiting on other rank data\n";
        MPI_Bcast(flatProportions, flatSize * imgCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
        std::cout << "rank " << rank << ": received data on other ranks\n";

        // Rebuild data
        auto weights = BuildWeights(flatProportions, hist, imgCount, rank);

        // Calculate scores
        auto scores = GetScores(weights, imgCount, rank);
        std::cout << "rank " << rank << ": calculated scores\n";

        MPI_Gather(scores, imgCount, MPI_FLOAT, nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD); // sending 1 x imgCount

        // TODO: delete objects
        std::cout << "rank " << rank << ": Finished\n";
    }

    MPI_Finalize();
    return 0;

}
