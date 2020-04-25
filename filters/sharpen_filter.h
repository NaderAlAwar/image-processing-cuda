#ifndef SHARPEN_FILTER_H
#define SHARPEN_FILTER_H

#include "../image.h"
#include "util.h"

stbi_uc* sharpen(stbi_uc* input_image, int width, int height, int channels);
__global__ void sharpen(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int padded_width, int padded_height, stbi_uc* mask, int mask_size);

stbi_uc* sharpen(stbi_uc* input_image, int width, int height, int channels) {
    stbi_uc mask[] = {126, 126, 126,
                      126, 135, 126,
                      126, 126, 126};
    int mask_size = 3;
    int* d_mask;

    cudaMallocManaged(&d_mask, mask_size * mask_size * sizeof(stbi_uc));
    cudaMemcpy(d_mask, mask, mask_size * mask_size * sizeof(stbi_uc), cudaMemcpyHostToDevice);

    int padded_width = width;
    int padded_height = height;
    stbi_uc* padded_image = zeroPadImage(input_image, padded_width, padded_height, channels, mask_size);

    int image_size = channels * padded_width * padded_height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, padded_image, image_size, cudaMemcpyHostToDevice);
    imageFree(padded_image);

    int total_threads = width * height;
    int threads = min(MAX_THREADS, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / MAX_THREADS;

    printf("Blocks %d, threads %d\n", blocks, threads);
    sharpen<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, padded_width, padded_height, total_threads, d_mask, mask_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void sharpen(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int padded_width, int padded_height, int total_threads, stbi_uc* mask, int mask_size) {
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= total_threads) {
        return;
    }

    int padding_size = mask_size / 2;

    int y_coordinate = (thread_id / width) + mask_size / 2;
    int x_coordinate = (thread_id % height) + mask_size / 2;
    Pixel pixel;
    getPixel(input_image, padded_width, x_coordinate, y_coordinate, &pixel);

    Pixel current_pixel;
    stbi_uc red = 0;
    stbi_uc blue = 0;
    stbi_uc green = 0;
    for (int i = 0; i < mask_size; i++) {
        for (int j = 0; j < mask_size; j++) {
            getPixel(input_image, padded_width, x_coordinate - padding_size + i, y_coordinate - padding_size + j, &current_pixel);
            int mask_element = mask[i * mask_size + j];

            red += current_pixel.r * mask_element;
            green += current_pixel.g * mask_element;
            blue += current_pixel.b * mask_element;
        }
    }

    pixel.r = (stbi_uc) red;
    pixel.g = (stbi_uc) green;
    pixel.b = (stbi_uc) blue;
}

#endif