#include <list>
#include <string>

#include "ImageReader.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "image.jpeg image2.jpeg ...\n";
        exit(1);
    }

    // Read in image file names
    std::list<std::string> images;
    for (int i = 1; i < argc; i++) {
        images.push_back(argv[i]);
    }

}