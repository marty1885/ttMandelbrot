#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <png.h>
#include <string>
#include "stb_image_write.h"

inline void map_color(float iteration_fraction, uint8_t* color) {
    // Control points for the Ultra Fractal color mapping
    struct ControlPoint {
        float position;
        uint8_t r, g, b;
    };

    ControlPoint control_points[] = {
        {0.0f, 0, 7, 100},
        {0.16f, 32, 107, 203},
        {0.42f, 237, 255, 255},
        {0.6425f, 255, 170, 0},
        {0.8575f, 0, 2, 0}
    };

    // Clamp iteration_fraction to the range [0, 1]
    if (iteration_fraction <= 0.0f) {
        color[0] = control_points[0].r;
        color[1] = control_points[0].g;
        color[2] = control_points[0].b;
        return;
    }
    if (iteration_fraction >= 1.0f) {
        color[0] = control_points[4].r;
        color[1] = control_points[4].g;
        color[2] = control_points[4].b;
        return;
    }

    // Find the two control points surrounding the iteration_fraction
    int i = 0;
    while (i < 4 && iteration_fraction > control_points[i + 1].position) {
        i++;
    }

    // Perform cubic interpolation
    auto interpolate = [](float t, float p0, float p1, float p2, float p3) -> float {
        float a = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
        float b = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
        float c = -0.5f * p0 + 0.5f * p2;
        float d = p1;
        return ((a * t + b) * t + c) * t + d;
    };

    float t = (iteration_fraction - control_points[i].position) /
              (control_points[i + 1].position - control_points[i].position);

    int p0 = (i > 0) ? i - 1 : 0;
    int p1 = i;
    int p2 = i + 1;
    int p3 = (i < 3) ? i + 2 : 4;

    color[0] = static_cast<uint8_t>(std::clamp(interpolate(t, control_points[p0].r, control_points[p1].r, control_points[p2].r, control_points[p3].r), 0.0f, 255.0f));
    color[1] = static_cast<uint8_t>(std::clamp(interpolate(t, control_points[p0].g, control_points[p1].g, control_points[p2].g, control_points[p3].g), 0.0f, 255.0f));
    color[2] = static_cast<uint8_t>(std::clamp(interpolate(t, control_points[p0].b, control_points[p1].b, control_points[p2].b, control_points[p3].b), 0.0f, 255.0f));
}

static bool write_png(const char* path, int width, int height, int channels, const uint8_t* rgb, int stride) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return false;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        fclose(fp);
        return false;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; ++y) {
        row_pointers[y] = const_cast<uint8_t*>(rgb + y * stride);
    }

    png_write_image(png_ptr, row_pointers.data());
    png_write_end(png_ptr, nullptr);

    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return true;
}

static bool save_image(const std::string& path, int width, int height, int channels, const uint8_t* rgb, int stride) {
    if(path.ends_with(".png")) {
        return write_png(path.c_str(), width, height, channels, rgb, stride);
    }
    else if(path.ends_with(".jpg")) {
        return stbi_write_jpg(path.c_str(), width, height, channels, rgb, 100);
    }
    else if(path.ends_with(".bmp") || path.find(".") == std::string::npos) {
        return stbi_write_bmp(path.c_str(), width, height, channels, rgb);
    }
    return false;
}
