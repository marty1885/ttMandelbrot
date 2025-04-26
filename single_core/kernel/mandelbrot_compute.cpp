#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#define ITERATIONS (8)

#ifdef TRISC_MATH
inline void mandelbrot(const uint dst_offset) {
  constexpr uint dst_tile_size = 32;
  for(int _=0;_<ITERATIONS;_++) {
    vFloat real = dst_reg[0];
    vFloat imag = dst_reg[dst_offset * dst_tile_size];
    vFloat zx = real;
    vFloat zy = imag;
    vFloat count = 0;

    constexpr int max_iter = 64;
    for(int i=0;i<max_iter;i++) {
      v_if(zx * zx + zy * zy < 4.f) {
        vFloat tmp = zx * zx - zy * zy + real;
        zy = 2.f * zx * zy + imag;
        zx = tmp;
        count += 1.f;
      }
      v_endif;
    }
    dst_reg[0] = count;
    dst_reg++;
  }
}
#endif

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    init_sfpu(cb_in0, cb_out0);
    add_binary_tile_init();
    for (uint32_t i = 0; i < n_tiles; i++) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile(cb_in0, 0, 0);
            copy_tile(cb_in1, 0, 1);
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);

            // pseudo code: dst[0] = mandelbrot(dst[0], dst[1]);
            MATH(llk_math_eltwise_binary_sfpu_params<false>(mandelbrot, 0, 1, VectorMode::RC);)

            cb_reserve_back(cb_out0, 1);
            pack_tile(0, cb_out0);
            tile_regs_commit();
            tile_regs_release();
            cb_push_back(cb_out0, 1);
    }
}
}
