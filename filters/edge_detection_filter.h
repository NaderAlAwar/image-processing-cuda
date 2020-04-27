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

stbi_uc** edgeDetectionBatchStreams(stbi_uc** input_images, int input_size, int width, int height, int channels) {
    int mask_x[] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};
    int mask_y[] = {1,  2,  1,
                    0,  0,  0,
                   -1, -2, -1};

    int mask_dimension = 3;
    Memory memory = Global;

    stbi_uc** output_images = convolveBatch(input_images, input_size, width, height, channels, mask_x, mask_dimension, memory, true);
    output_images = convolveBatch(output_images, input_size, width, height, channels, mask_y, mask_dimension, memory, true);

    return output_images;
}

stbi_uc** edgeDetectionBatchSequential(stbi_uc** input_images, int input_size, int width, int height, int channels) {
    int mask_x[] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};
    int mask_y[] = {1,  2,  1,
                    0,  0,  0,
                   -1, -2, -1};

    int mask_dimension = 3;
    Memory memory = Global;

    stbi_uc** output_images = convolveBatch(input_images, input_size, width, height, channels, mask_x, mask_dimension, memory, false);
    output_images = convolveBatch(output_images, input_size, width, height, channels, mask_y, mask_dimension, memory, false);

    return output_images;
}

#endif