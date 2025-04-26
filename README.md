# ttMandelbrot - Mandelbrot Set renderer using Tenstorrent hardware

This program has only been tested on Tenstorrent Wormhole. It WILL NOT work on Grayskull and may not work on Blackhole.

## Build and use

### Dependencies
- tt-Metalium SDK
- OpenMP (for CPU reference)
- libpng (for image saving)

### Build

```bash
export TT_METAL_HOME=/path/to/tt-metal
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

### Usage

There will be 4 files generated. **Run them in the build directory** else the kernel files will not be found.

* `cpu` - CPU reference implementation
* `tt_single_core` - Baseline single (Tensix) core implementation using DRAM to store initial real and imaginary parts of the complex number
* `tt_single_core_nullary` - Baseline single (Tensix) core implementation but the complex number is generated on the fly
* `tt_multi_core_nullary` - Optimized multi-core implementation version of the above

Each support a set of common parameters:
- `--width <width>` - Width of the image in pixels
- `--height <height>` - Height of the image in pixels
- `--output <output>` - Output file name (Supported formats: PNG, JPEG, BMP)
- `--help` - Display the program help message

For details please refer to the help message.

### Benchmarking

`benchmark.sh` can be run from the root of the project after building to benchmark the performance of the different implementations. The results will be saved in `benchmark.csv`. It must be ran using zsh.

```bash
zsh benchmark.sh
```
