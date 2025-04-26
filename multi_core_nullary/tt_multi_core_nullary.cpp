
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/persistent_kernel_cache.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include <chrono>

#include "stb_image_write.h"
#include "utils.hpp"

using namespace tt::tt_metal;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, bool sram = false) {
    InterleavedBufferConfig config{
        .device = device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)};
    return CreateBuffer(config);
}

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t n_tiles, size_t element_size, bool sram = false) {
    const uint32_t tile_size = element_size * TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

CBHandle MakeCircularBufferFP32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(float) * TILE_WIDTH * TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float32);
}

std::string next_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Expected argument after " << argv[i] << std::endl;
        exit(1);
    }
    return argv[++i];
}

void help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "This program demonstrates how to add two vectors using tt-Metalium.\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --device, -d <device_id>   Specify the device to run the program on. Default is 0.\n";
    std::cout << "  --seed, -s <seed>          Specify the seed for the random number generator. Default is random.\n";
    std::cout << "  --width, -w <width>        Specify the width of the image. Default is 1024.\n";
    std::cout << "  --height, -h <height>      Specify the height of the image. Default is 1024.\n";
    std::cout << "  --output, -o <output_file> Specify the output file name. Default is mandelbrot_tt_multi_core_nullary.png.\n";
    std::cout << "                             Supported formats: PNG, JPG, BMP.\n";
    std::cout << "  --help                     Display this help message.\n";
    exit(0);
}

int main(int argc, char** argv) {
    int seed = std::random_device{}();
    int device_id = 0;
    size_t width = 1024;
    size_t height = 1024;
    std::string output_file = "mandelbrot_tt_multi_core_nullary.png";

    // Quick and dirty argument parsing.
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        if (arg == "--device" || arg == "-d") {
            device_id = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--seed" || arg == "-s") {
            seed = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--width" || arg == "-w") {
            width = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--height" || arg == "-h") {
            height = std::stoi(next_arg(i, argc, argv));
        } else if (arg == "--output" || arg == "-o") {
            output_file = next_arg(i, argc, argv);
        } else if (arg == "--help" || arg == "-h") {
            help(argv[0]);
            return 0;
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
            help(argv[0]);
        }
    }

    // tt::tt_metal::detail::EnablePersistentKernelCache();
    IDevice* device = CreateDevice(device_id);
    device->enable_program_cache();

    Program program = CreateProgram();
    auto core_grid = device->compute_with_storage_grid_size();
    auto all_cores = CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1});

    CommandQueue& cq = device->command_queue();

    const float left = -2.0f;
    const float right = 1.0f;
    const float bottom = -1.5f;
    const float top = 1.5f;

    const uint32_t tile_size = TILE_WIDTH * TILE_HEIGHT;
    if(width % tile_size != 0)
        throw std::runtime_error("Invalid dimensions, width must be divisible by tile_size");
    const uint32_t n_tiles = (width * height) / tile_size;
    auto c = MakeBuffer(device, n_tiles, sizeof(float));

    const uint32_t tiles_per_cb = 4;
    CBHandle cb_a = MakeCircularBufferFP32(program, all_cores, tt::CBIndex::c_0, tiles_per_cb); // Why???
    // CBHandle cb_b = MakeCircularBufferFP32(program, core, tt::CBIndex::c_1, tiles_per_cb);
    CBHandle cb_c = MakeCircularBufferFP32(program, all_cores, tt::CBIndex::c_16, tiles_per_cb);

    auto writer = CreateKernel(
        program,
        "../multi_core_nullary/kernel/tile_write.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto compute = CreateKernel(
        program,
        "../multi_core_nullary/kernel/mandelbrot_compute.cpp",
        all_cores,
        ComputeConfig{.math_approx_mode = false, .compile_args = {}, .defines = {}});

    // SetRuntimeArgs(program, reader, core, {a->address(), b->address(), n_tiles});
    uint32_t params[4];
    float p[4] = {left, right, bottom, top};
    memcpy(params, p, sizeof(p));

    uint32_t num_cores = core_grid.x * core_grid.y;
    for(uint32_t y = 0; y < core_grid.y; y++) {
        for(uint32_t x = 0; x < core_grid.x; x++) {
            auto core = CoreCoord(x, y);
            uint32_t core_id = core.x + core.y * core_grid.x;
            uint32_t start_row = height / num_cores * core_id;
            uint32_t end_row = start_row + height / num_cores;
            if(end_row > height)
                end_row = height;
            if(core_id == (core_grid.y - 1) * (core_grid.x - 1))
                end_row = height;
            SetRuntimeArgs(program, writer, core, {c->address(), start_row, end_row, uint32_t(width / tile_size)});
            SetRuntimeArgs(program, compute, core, {params[0], params[1], params[2], params[3], uint32_t(width), uint32_t(height), start_row, end_row});
        }
    }

    Finish(cq);
    EnqueueProgram(cq, program, true); // Run it a 1st time to get the compiler out of the way
    auto start = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    std::vector<float> c_data;
    EnqueueReadBuffer(cq, c, c_data, true);
    float* c_bf16 = reinterpret_cast<float*>(c_data.data());

    std::vector<uint8_t> image(width * height * 3);
    #pragma omp parallel for
    for(size_t y = 0; y < height; ++y) {
        for(size_t x = 0; x < width; ++x) {
            float iteration = c_bf16[y * width + x];
            constexpr int max_iteration = 64;
            map_color(iteration/max_iteration, image.data() + y * width * 3 + x * 3);
        }
    }
    if(!save_image("mandelbrot_tt_multi_core_nullary.png", width, height, 3, image.data(), width * 3)) {
        std::cerr << "Failed to save image." << std::endl;
    }

    // Finally, we close the device.
    CloseDevice(device);
    return 0;
}
