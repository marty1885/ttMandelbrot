#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#define ITERATIONS (8)

#ifdef TRISC_MATH
inline void mandelbrot(float y_coord, float left, float right) {
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(0);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    float lane_delta = (right - left) / 1024.f;
    constexpr int max_iter = 64;
    float lane_offset = 0.0f;
    // HACK: Split the even and odd iterations because SFPU does interleaved foramt (for some reason)
    for(int i=0;i<32;i+=2) {
        vFloat vleft = left;
        vFloat vright = right;
        vFloat vi = int32_to_float(i / 2);
        vFloat vdelta = vright - vleft;
        vFloat vchunk_coarse = vleft + (vdelta) * (1.f / 16.f) * vi;
        vFloat vreal_lane_id = int32_to_float(vConstTileId);
        vFloat vchunk_fine = vchunk_coarse + (vreal_lane_id) * lane_delta;

        vFloat real = vchunk_fine;
        vFloat imag = y_coord;
        vFloat zx = real;
        vFloat zy = imag;
        vFloat count = 0;

        for(int i=0;i<max_iter;i++) {
          v_if(zx * zx + zy * zy < 4.f) {
            vFloat tmp = zx * zx - zy * zy + real;
            zy = 2.f * zx * zy + imag;
            zx = tmp;
            count += 1.f;
          }
          v_endif;
        }
        dst_reg[i] = count;
    }

    for(int i=1;i<32;i+=2) {
        vFloat vleft = left;
        vFloat vright = right;
        vFloat vi = int32_to_float(i / 2);
        vFloat vdelta = vright - vleft;
        vFloat vchunk_coarse = vleft + (vdelta) * (1.f / 16.f) * vi;
        vFloat vreal_lane_id = int32_to_float(vConstTileId+1);
        vFloat vchunk_fine = vchunk_coarse + (vreal_lane_id) * lane_delta;

        vFloat real = vchunk_fine;
        vFloat imag = y_coord;
        vFloat zx = real;
        vFloat zy = imag;
        vFloat count = 0;

        for(int i=0;i<max_iter;i++) {
          v_if(zx * zx + zy * zy < 4.f) {
            vFloat tmp = zx * zx - zy * zy + real;
            zy = 2.f * zx * zy + imag;
            zx = tmp;
            count += 1.f;
          }
          v_endif;
        }
        dst_reg[i] = count;
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

#endif

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t TILE_HEIGHT = 32;
    float left = get_arg_val<float>(0);
    float right = get_arg_val<float>(1);
    float bottom = get_arg_val<float>(2);
    float top = get_arg_val<float>(3);
    uint32_t width = get_arg_val<uint32_t>(4);
    uint32_t height = get_arg_val<uint32_t>(5);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    init_sfpu(cb_in0, cb_out0);
    add_binary_tile_init();
    for(uint32_t y = 0; y < height; y++) {
        float y_coord = bottom + (top - bottom) * y / height;
        for(uint32_t x = 0; x < width / (TILE_WIDTH * TILE_HEIGHT); x += 1) {
            tile_regs_acquire();
            tile_regs_wait();
            uint32_t x_max = width / (TILE_WIDTH * TILE_HEIGHT);
            float l = left + (right - left) * x / x_max;
            float r = l + (right - left) / x_max;

            // pseudo code: dst[0] = mandelbrot(y_coord, l, r);
            MATH(mandelbrot(y_coord, l, r));

            cb_reserve_back(cb_out0, 1);
            pack_tile(0, cb_out0);
            tile_regs_commit();
            tile_regs_release();
            cb_push_back(cb_out0, 1);
        }
    }
}
}
