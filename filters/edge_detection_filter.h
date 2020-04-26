#ifndef EDGE_DETECTION_FILTER_H
#define EDGE_DETECTION_FILTER_H

#include "../image.h"
#include "util.h"

stbi_uc* edgeDetection(stbi_uc* input_image, int width, int height, int channels);
__global__ void edgeDetection(const stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int padded_width, int padded_height, int* mask, int mask_size);
__global__ void edgeDetection2(const stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int padded_width, int padded_height, int* mask, int mask_size);

stbi_uc* edgeDetection(stbi_uc* input_image, int width, int height, int channels) {
    int mask_x[] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};
    int mask_y[] = {1,  2,  1,
                    0,  0,  0,
                   -1, -2, -1};
    int mask_size = 3;
    int* d_mask_x;
    int* d_mask_y;

    cudaMallocManaged(&d_mask_x, mask_size * mask_size * sizeof(int));
    cudaMallocManaged(&d_mask_y, mask_size * mask_size * sizeof(int));
    cudaMemcpy(d_mask_x, mask_x, mask_size * mask_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask_y, mask_y, mask_size * mask_size * sizeof(int), cudaMemcpyHostToDevice);

    int padded_width = width;
    int padded_height = height;
    stbi_uc* padded_image = zeroPadImage(input_image, padded_width, padded_height, channels, mask_size);

    int image_size = channels * padded_width * padded_height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image_x;
    stbi_uc* d_output_image_y;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);
    for (int i = 0; i < padded_width * padded_height; i++) {
        h_output_image[i] = input_image[i];
    }

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image_x, image_size);
    cudaMallocManaged(&d_output_image_y, image_size);
    cudaMemcpy(d_input_image, padded_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_image_x, padded_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_image_y, padded_image, image_size, cudaMemcpyHostToDevice);
    imageFree(padded_image);

    int total_threads = width * height;
    int threads = min(MAX_THREADS, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / MAX_THREADS;

    printf("Blocks %d, threads %d, total threads %d\n", blocks, threads, total_threads);
    edgeDetection<<<blocks, threads>>>(d_input_image, d_output_image_x, width, height, channels, total_threads, padded_width, padded_height, d_mask_x, mask_size);
    cudaDeviceSynchronize();
    edgeDetection<<<blocks, threads>>>(d_output_image_x, d_output_image_y, padded_width, padded_height, channels, total_threads, padded_width, padded_height, d_mask_y, mask_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image_y, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void edgeDetection(const stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int padded_width, int padded_height, int* mask, int mask_size) {
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= total_threads) {
        return;
    }
    
    int padding_size = mask_size / 2;

    int y_coordinate = (thread_id / width) + padding_size;
    int x_coordinate = (thread_id % height) + padding_size;

    Pixel current_pixel;
    int red = 0;
    int blue = 0;
    int green = 0;
    int alpha = 0;
    for (int i = 0; i < mask_size; i++) {
        for (int j = 0; j < mask_size; j++) {
            getPixel(input_image, padded_width, x_coordinate - padding_size + i, y_coordinate - padding_size + j, &current_pixel);
            int mask_element = mask[i * mask_size + j];

            red += current_pixel.r * mask_element;
            green += current_pixel.g * mask_element;
            blue += current_pixel.b * mask_element;
            alpha += current_pixel.a * mask_element;
        }
    }

    Pixel pixel;
    getPixel(input_image, padded_width, x_coordinate, y_coordinate, &pixel);
    if (red < 0) {
        pixel.r = 0;
    } else if (red > 255) {
        pixel.r = 255;
    } else {
        pixel.r = red;
    }
    if (green < 0) {
        pixel.g = 0;
    } else if (green > 255) {
        pixel.g = 255;
    } else {
        pixel.g = green;
    }
    if (blue < 0) {
        pixel.b = 0;
    } else if (blue > 255) {
        pixel.b = 255;
    } else {
        pixel.b = blue;
    }

    setPixel(output_image, width, x_coordinate, y_coordinate, &pixel);
}

#endif