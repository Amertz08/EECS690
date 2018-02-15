#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include <mutex>
#include <thread>
#include "Barrier.hpp"

std::mutex*** tracks;
std::thread** threads;
std::mutex print_mutex;
int* stepCount;
Barrier b;

/**
 * Thread safe print function
 * @param text
 */
void thread_print(std::string text)
{
    print_mutex.lock();
    std::cout << text;
    print_mutex.unlock();
}

bool go = false;

/**
 * Runs trains
 * @param trainID
 * @param barrierCount
 * @param moves : vector of moves
 */
void runner(int trainID, int barrierCount, std::vector<int>* moves)
{
    b.barrier(barrierCount);
    while (!go)
        ;
    thread_print("Train: " + std::to_string(trainID) + " bcount: " + std::to_string(barrierCount) + "\n");
    for (int i = 0; i < moves->size() - 1; i++) {
        int current = moves->at(i);
        int next = moves->at(i + 1);
        int a, b;

        // Set a to smallest of pair for easier reading
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

        if (tracks[current][next]->try_lock()) {
            message += "\n";
            thread_print(message);
            tracks[current][next]->unlock();
        } else {
            message += " must stay at station " + std::to_string(current) + "\n";
            thread_print(message);
            i--; // Move isn't actually made so decrement
        }
        stepCount[trainID]++;
    }
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
    }

    int nTrains, nStations;
    inputFile >> nTrains;
    inputFile >> nStations;
    std::cout << "nTrains: " << nTrains << " nStations: " << nStations << std::endl;


    // Create Lists
    auto trains = new std::vector<int>[nTrains];
    threads = new std::thread*[nTrains];
    stepCount = new int[nTrains]();

    // Load lists
    for (int i = 0; i < nTrains; i++) {
        int stationCount;
        inputFile >> stationCount;
        std::cout << "Train: " << i << " Inserting " << stationCount << " stations\n";
        for (int j = 0; j < stationCount; j++) {
            int val;
            inputFile >> val;
            std::cout << val << " ";
            trains[i].push_back(val);
        }
        std::cout << std::endl;
        // function, trainID, barrierCount, moves
        threads[i] = new std::thread(runner, i, nTrains, &trains[i]);
    }
    inputFile.close();
    // End file read

    // Create mutexes
    tracks = new std::mutex**[nStations];
    for (int i = 0; i < nStations; i++)
        tracks[i] = new std::mutex*[nStations];

    // Copy mutexes
    for (int i = 0; i < nStations; i++) {
        for (int j = i; j < nStations; j++) {
            if (j == i)
                tracks[i][j] = nullptr;
            else {
                tracks[i][j] = new std::mutex;
                tracks[j][i] = tracks[i][j];
            }
        }
    }

    std::cout << "Running trains\n";
    go = true;
    for (int i = 0; i < nTrains; i++)
        threads[i]->join();
    std::cout << "Ending simulation\n";

    // Delete mutexes
    for (int i = 0; i < nStations; i++) {
        for (int j = i; j < nStations; j++) {
            if (i != j)
                delete tracks[i][j];
        }
        delete[] tracks[i];
    }
    delete[] tracks;
    delete[] trains;
    delete[] stepCount;

    return 0;
}