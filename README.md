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
