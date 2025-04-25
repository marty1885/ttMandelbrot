#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

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
