#ifndef EDGE_DETECTION_FILTER_H
#define EDGE_DETECTION_FILTER_H

#include "../image.h"
#include "convolve.h"

stbi_uc* edgeDetection(stbi_uc* input_image, int width, int height, int channels) {
    int mask_x[] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};
    int mask_y[] = {1,  2,  1,
                    0,  0,  0,
                   -1, -2, -1};

    int mask_dimension = 3;
    Memory memory = Global;

    stbi_uc* output = convolve(input_image, width, height, channels, mask_x, mask_dimension, memory);
    output = convolve(output, width, height, channels, mask_y, mask_dimension, memory);

    return output;
}

stbi_uc* edgeDetectionSharedMemory(stbi_uc* input_image, int width, int height, int channels) {
    int mask_x[] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};
    int mask_y[] = {1,  2,  1,
                    0,  0,  0,
                   -1, -2, -1};

    int mask_dimension = 3;
    Memory memory = Shared;

    stbi_uc* output = convolve(input_image, width, height, channels, mask_x, mask_dimension, memory);
    output = convolve(output, width, height, channels, mask_y, mask_dimension, memory);

    return output;
}

stbi_uc* edgeDetectionConstantMemory(stbi_uc* input_image, int width, int height, int channels) {
    int mask_x[] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};
    int mask_y[] = {1,  2,  1,
                    0,  0,  0,
                   -1, -2, -1};

    int mask_dimension = 3;
    Memory memory = Constant;

    stbi_uc* output = convolve(input_image, width, height, channels, mask_x, mask_dimension, memory);
    output = convolve(output, width, height, channels, mask_y, mask_dimension, memory);

    return output;
}

stbi_uc* edgeDetectionTextureMemory(stbi_uc* input_image, int width, int height, int channels) {
    int mask_x[] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};
    int mask_y[] = {1,  2,  1,
                    0,  0,  0,
                   -1, -2, -1};

    int mask_dimension = 3;
    Memory memory = Texture;

    stbi_uc* output = convolve(input_image, width, height, channels, mask_x, mask_dimension, memory);
    output = convolve(output, width, height, channels, mask_y, mask_dimension, memory);

    return output;
}

#endif