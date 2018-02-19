#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include "Barrier.hpp"

std::atomic_flag*** tracks;
std::thread** threads;
std::mutex print_mutex;
int* stepCount;

int barrierCount;
int nextBarrierCount;
std::mutex barrier_mutex;

Barrier stepBarrier; // Syncs thread steps
bool go = false;

/**
 * Thread safe print function
 * @param text
 */
void thread_print(const std::string &text)
{
    print_mutex.lock();
    std::cout << text;
    print_mutex.unlock();
}

/**
 * Runs trains
 * @param trainID
 * @param moves : vector of moves
 */
void runner(int trainID, std::vector<int>* moves)
{
    while (!go)
        ;
    for (unsigned long i = 0; i < moves->size() - 1; i++) {
        int current = moves->at(i);
        int next = moves->at(i + 1);
        int a, b;

        // Set a to smallest of pair for easier reading in later print statement
        if (current > next) {
            a = next;
            b = current;
        } else {
            a = current;
            b = next;
        }

        std::string message = "step: " + std::to_string(stepCount[trainID]) + " train: " + std::to_string(trainID);
        message += " (" + std::to_string(current) + " -> " + std::to_string(next) + ")";
        message += " (" + std::to_string(a) + ", " + std::to_string(b) + ")";

        stepBarrier.barrier(barrierCount);
        if (!tracks[current][next]->test_and_set(std::memory_order_acquire)) {
            message += "\n";
            thread_print(message);
            tracks[current][next]->clear(std::memory_order_release);
        } else {
            message += " must stay at station " + std::to_string(current) + "\n";
            thread_print(message);
            i--; // Move isn't actually made so decrement
        }
        stepCount[trainID]++;
        if (barrierCount != nextBarrierCount) {
            barrier_mutex.lock();
            barrierCount = nextBarrierCount;
            barrier_mutex.unlock();
        }
    }
    barrier_mutex.lock();
    nextBarrierCount--;
    barrier_mutex.unlock();
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << argv[0] << " INPUT_FILE\n";
        exit(1);
    }

    std::string fileName = argv[1];
    std::ifstream inputFile;

    inputFile.open(fileName);

    if (!inputFile) {
        std::cerr << fileName << " failed to open properly\n";
        exit(1);
    }

    int nTrains, nStations;
    inputFile >> nTrains;
    inputFile >> nStations;
    std::cout << "nTrains: " << nTrains << " nStations: " << nStations << std::endl;

    barrierCount = nTrains;
    nextBarrierCount = nTrains;

    // Create Lists
    auto trains = new std::vector<int>[nTrains];
    threads = new std::thread*[nTrains];
    stepCount = new int[nTrains]();

    // Load lists
    for (int i = 0; i < nTrains; i++) {
        int stationCount;
        inputFile >> stationCount;
        std::cout << "Train: " << i << " Inserting " << stationCount << " stations ";
        for (int j = 0; j < stationCount; j++) {
            int val;
            inputFile >> val;
            std::cout << val << " ";
            trains[i].push_back(val);
        }
        std::cout << std::endl;
        // function, trainID, barrierCount, moves
        threads[i] = new std::thread(runner, i, &trains[i]);
    }
    inputFile.close();
    // End file read

    // Create 2D array
    tracks = new std::atomic_flag**[nStations];
    for (int i = 0; i < nStations; i++)
        tracks[i] = new std::atomic_flag*[nStations];

    // Create mutexes
    for (int i = 0; i < nStations; i++) {
        for (int j = i; j < nStations; j++) {
            if (j == i)
                tracks[i][j] = nullptr;
            else {
                auto f = new std::atomic_flag;
                f->clear();
                tracks[i][j] = f;
                tracks[j][i] = tracks[i][j];
            }
        }
    }

    std::cout << "Running trains\n";
    go = true;
    for (int i = 0; i < nTrains; i++)
        threads[i]->join();
    std::cout << "Ending simulation\n";

    for (int i = 0; i < nTrains; i++)
        std::cout << "Train: " << i << " finished in " << stepCount[i] << " steps\n";

    // Delete mutexes
    for (int i = 0; i < nStations; i++) {
        for (int j = i; j < nStations; j++) {
            if (i != j)
                delete tracks[i][j];
        }
        delete[] tracks[i];
    }
    delete[] tracks;

    // Delete threads
    for (int i = 0; i < nTrains; i++)
        delete threads[i];
    delete[] threads;

    // Delete everything else
    delete[] trains;
    delete[] stepCount;

    return 0;
}