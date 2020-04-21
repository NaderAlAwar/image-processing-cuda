#ifndef HORIZONTAL_FLIP_FILTER_H
#define HORIZONTAL_FLIP_FILTER_H

#include "../image.h"

const int MAX_THREADS = 1024;

stbi_uc* horizontalFlip(stbi_uc* input_image, int width, int height, int channels);
__global__ void horizontalFlip(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);

stbi_uc* horizontalFlip(stbi_uc* input_image, int width, int height, int channels) {
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = width * height / 2;
    int threads = min(MAX_THREADS, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / MAX_THREADS;

    printf("Blocks %d, threads %d\n", blocks, threads);
    horizontalFlip<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, total_threads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void horizontalFlip(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads) {
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= total_threads) {
        return;
    }

    int y_coordinate = thread_id / (width / 2);
    int x_coordinate_1 = thread_id % height;
    int x_coordinate_2 = width - x_coordinate_1;
    Pixel leftPixel, rightPixel;

    printf("Height = %d\n", height);
    printf("Width = %d\n", width);

    getPixel(input_image, width, x_coordinate_1, y_coordinate, &leftPixel);
    getPixel(input_image, width, x_coordinate_2, y_coordinate, &rightPixel);

    setPixel(output_image, width, x_coordinate_1, y_coordinate, &rightPixel);
    setPixel(output_image, width, x_coordinate_2, y_coordinate, &leftPixel);
}

#endif