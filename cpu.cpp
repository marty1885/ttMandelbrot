#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <omp.h>
#include <thread>

#include "stb_image_write.h"
#include "utils.hpp"

void help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "This program demonstrates how to add two vectors using tt-Metalium.\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --width, -w <width>        Specify the width of the image. Default is 1024.\n";
    std::cout << "  --height, -h <height>      Specify the height of the image. Default is 1024.\n";
    std::cout << "  --output, -o <filename>    Specify the output filename. Default is mandelbrot.png.\n";
    std::cout << "                             Supported formats: PNG, JPG, BMP.\n";
    std::cout << "  --threads, -t <num_threads> Specify the number of threads to use. Default is auto.\n";
    std::cout << "  --help                     Display this help message.\n";
    exit(0);
}

std::string next_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Expected argument after " << argv[i] << std::endl;
        exit(1);
    }
    return argv[++i];
}

int main(int argc, char* argv[])
{
    size_t width = 1024;
    size_t height = 1024;

    float left = -2.0f;
    float right = 1.0f;
    float bottom = -1.5f;
    float top = 1.5f;
    std::string output_file = "mandelbrot.png";
    int n_threads = std::thread::hardware_concurrency();

    const int max_iteration = 64;

    // Quick and dirty argument parsing.
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "--width" || arg == "-w") {
            width = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--height" || arg == "-h") {
            height = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--output" || arg == "-o") {
            output_file = next_arg(i, argc, argv);
        } else if (arg == "--threads" || arg == "-t") {
            n_threads = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--help") {
            help(argv[0]);
            return 0;
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
            help(argv[0]);
        }
    }


    std::vector<int> iterations(width * height);
    omp_set_num_threads(n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for(size_t y = 0; y < height; ++y) {
        for(size_t x = 0; x < width; ++x) {
            float real = left + (right - left) * x / (width - 1);
            float imag = bottom + (top - bottom) * y / (height - 1);

            float zx = real;
            float zy = imag;
            int iteration = 0;
            while(zx * zx + zy * zy < 4.0f && iteration < max_iteration) {
                float tmp = zx * zx - zy * zy + real;
                zy = 2.0f * zx * zy + imag;
                zx = tmp;
                ++iteration;
            }

            iterations[y * width + x] = iteration;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    std::vector<uint8_t> image(width * height * 3);
    omp_set_num_threads(std::thread::hardware_concurrency());
    #pragma omp parallel for
    for(size_t y = 0; y < height; ++y) {
        for(size_t x = 0; x < width; ++x) {
            int iteration = iterations[y * width + x];
            int max_iteration = 64;
            map_color((float)iteration/max_iteration, image.data() + y * width * 3 + x * 3);
        }
    }

    // Save the image
    if(!save_image(output_file, width, height, 3, image.data(), width * 3)) {
        std::cerr << "Failed to save image." << std::endl;
    }

}
