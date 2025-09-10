# Claude Code Session Notes

## Fixed Compilation Errors (2025-09-10)

### Issue
Compilation failed with missing header errors:
- `nlohmann/json.hpp: No such file or directory`
- `reflect: No such file or directory` 
- `enchantum/scoped.hpp: No such file or directory`
- `tt-logger/tt-logger.hpp: No such file or directory`

### Root Cause
CMake wasn't properly linking CPM (CMake Package Manager) dependencies that were already downloaded to `.cpmcache/`. The tt-metal library needed access to these dependencies' include paths.

### Solution
Fixed by properly linking CPM package targets instead of hard-coding hash-based directory paths:

1. **nlohmann/json**: Added `nlohmann_json::nlohmann_json` target
2. **enchantum**: Added `enchantum` target  
3. **boost-ext/reflect**: Added fallback path detection using `file(GLOB)` with directory validation
4. **tt-logger**: Added `tt-logger` target

### Key Changes in CMakeLists.txt
```cmake
target_link_libraries(ttmetal INTERFACE
    fmt
    yaml-cpp
    Boost::core
    Boost::container
    nlohmann_json::nlohmann_json
    enchantum
    tt-logger
    tt_metal
)

# Add reflect include directory (header-only library)
if(reflect_ADDED)
    target_include_directories(ttmetal INTERFACE ${reflect_SOURCE_DIR})
else()
    file(GLOB REFLECT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/.cpmcache/reflect/*/")
    foreach(REFLECT_DIR ${REFLECT_DIRS})
        if(IS_DIRECTORY ${REFLECT_DIR} AND EXISTS "${REFLECT_DIR}/reflect")
            target_include_directories(ttmetal INTERFACE ${REFLECT_DIR})
            break()
        endif()
    endforeach()
endif()
```

### Result
- ✅ All three executables compile successfully: `tt_single_core`, `tt_single_core_nullary`, `tt_multi_core_nullary`
- ✅ Scalable solution using CPM target system instead of hard-coded hash paths
- ✅ Proper dependency ordering (dependencies loaded before ttmetal target definition)

### Cost
- Total cost: $0.79
- Duration: 10m 2.5s
- Code changes: 60 lines added, 42 lines removed

## Fixed Kernel API Compatibility (2025-09-10)

### Issue
Runtime compilation failed for compute kernel with missing API functions:
- `'set_addr_mod_base' is not a member of 'ckernel::math'`
- `'clear_addr_mod_base' is not a member of 'ckernel::math'`

### Root Cause
The `set_addr_mod_base()` and `clear_addr_mod_base()` functions were removed from the Blackhole architecture API but are still required for Wormhole compatibility. This created an incompatibility when the same kernel code needed to run on both architectures.

### Solution
Added conditional compilation using architecture-specific macros:

```cpp
#ifdef ARCH_WORMHOLE
    math::set_addr_mod_base();
#endif

// ... kernel code ...

#ifdef ARCH_WORMHOLE
    math::clear_addr_mod_base();
#endif
```

### Key Changes in mandelbrot_compute.cpp:12,75
- Wrapped `math::set_addr_mod_base()` call in `#ifdef ARCH_WORMHOLE`
- Wrapped `math::clear_addr_mod_base()` call in `#ifdef ARCH_WORMHOLE`

### Result
- ✅ Maintains compatibility with both Wormhole and Blackhole architectures
- ✅ Kernel compiles and runs successfully on Blackhole devices  
- ✅ Preserves Wormhole functionality when needed
- ✅ Program executes successfully in 0.034 seconds

## Fixed Binary SFPU Function Signature (2025-09-10)

### Issue
Runtime compilation failed for `single_core` kernel with function signature mismatch:
- `error: too many arguments to function` in `_llk_math_eltwise_binary_sfpu_params_`
- The `mandelbrot` function expected only `dst_offset` but received multiple parameters

### Root Cause  
The binary SFPU API template `_llk_math_eltwise_binary_sfpu_params_` passes multiple parameters (`dst_index_in0`, `dst_index_in1`, `dst_index_out`) to the callback function, but the `mandelbrot` function signature was incorrect.

### Solution
Updated the `mandelbrot` function signature to match the binary SFPU API expectations:

```cpp
// Before:
inline void mandelbrot(const uint dst_offset)

// After:  
inline void mandelbrot(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out)
```

### Key Changes in single_core/kernel/mandelbrot_compute.cpp:11,14,30
- Updated function signature to accept three parameters instead of one
- Changed `dst_reg[0]` to `dst_reg[dst_index_in0]` for reading real component
- Changed `dst_reg[dst_offset * dst_tile_size]` to `dst_reg[dst_index_in1]` for reading imaginary component  
- Changed `dst_reg[0]` to `dst_reg[dst_index_out]` for writing result

### Additional Fixes Applied
Applied the same architecture-conditional compilation fix to `multi_core_nullary` kernel:
- Wrapped `math::set_addr_mod_base()` and `math::clear_addr_mod_base()` calls in `#ifdef ARCH_WORMHOLE`

### Result
- ✅ All three executables now compile and run successfully
- ✅ `tt_single_core`: 0.033 seconds execution time
- ✅ `tt_single_core_nullary`: 0.034 seconds execution time  
- ✅ `tt_multi_core_nullary`: 0.0004 seconds execution time
- ✅ Complete API compatibility between Wormhole and Blackhole architectures