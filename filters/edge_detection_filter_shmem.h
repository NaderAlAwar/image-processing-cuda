#ifndef EDGE_DETECTION_FILTER_SHMEM_H
#define EDGE_DETECTION_FILTER_SHMEM_H

#include "math.h"
#include "../image.h"
#include "util.h"

stbi_uc* edgeDetectionShmem(stbi_uc* input_image, int width, int height, int channels);
__global__ void edgeDetectionShmem(const stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int* mask, int mask_size, int shmem_width);
__device__ bool isOutOfBounds(int x, int y, int image_width, int image_height);

const int shmem_size = MAX_THREADS * 4 * sizeof(stbi_uc);

stbi_uc* edgeDetectionShmem(stbi_uc* input_image, int width, int height, int channels) {
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

    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image_x;
    stbi_uc* d_output_image_y;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);
    for (int i = 0; i < width * height * channels; i++) {
        h_output_image[i] = input_image[i];
    }

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image_x, image_size);
    cudaMallocManaged(&d_output_image_y, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_image_x, input_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_image_y, input_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = width * height;
    int threads = min(MAX_THREADS, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / MAX_THREADS;

    int square_root = (int) sqrt(threads);
    dim3 block(32, 32);
    dim3 grid;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (width + block.y - 1) / block.y;
    int shared_memory_size = MAX_THREADS * channels * sizeof(stbi_uc); 

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    printf("Blocks %d, threads %d, total threads %d\n", blocks, threads, total_threads);
    edgeDetectionShmem<<<grid, block, shared_memory_size>>>(d_input_image, d_output_image_x, width, height, channels, total_threads, d_mask_x, mask_size, square_root);
    cudaDeviceSynchronize();
    edgeDetectionShmem<<<grid, block, shared_memory_size>>>(d_output_image_x, d_output_image_y, width, height, channels, total_threads, d_mask_y, mask_size, square_root);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&time, start, stop);
    printf("Shared memory: %f\n", time);

    cudaMemcpy(h_output_image, d_output_image_y, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__device__ bool isOutOfBounds(int x, int y, int image_width, int image_height) {
    if (x < 0 || y < 0 || x >= image_width || y >= image_height) {
        return true;
    } else {
        return false;
    }
}

__global__ void edgeDetectionShmem(const stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int* mask, int mask_size, int shmem_width) {
    const int x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_coordinate = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ stbi_uc data[];
    int padding_size = mask_size / 2;
    int shmem_x = threadIdx.x;
    int shmem_y = threadIdx.y;

    Pixel current_pixel;
    getPixel(input_image, width, x_coordinate, y_coordinate, &current_pixel);
    setPixel(data, shmem_width, shmem_x, shmem_y, &current_pixel);
    __syncthreads();
    int red = 0;
    int blue = 0;
    int green = 0;
    int alpha = 0;
    for (int i = 0; i < mask_size; i++) {
        for (int j = 0; j < mask_size; j++) {
            int current_x_global = x_coordinate - padding_size + i;
            int current_y_global = y_coordinate - padding_size + j;
            int current_x = shmem_x - padding_size + i;
            int current_y = shmem_y - padding_size + j;
            if (isOutOfBounds(current_x_global, current_y_global, width, height)) {
                continue;
            }
            if (isOutOfBounds(current_x, current_y, shmem_width, shmem_width)) {
                getPixel(input_image, width, current_x_global, current_y_global, &current_pixel);
            } else {
                getPixel(data, shmem_width, current_x, current_y, &current_pixel);
            }
            
            int mask_element = mask[i * mask_size + j];

            red += current_pixel.r * mask_element;
            green += current_pixel.g * mask_element;
            blue += current_pixel.b * mask_element;
        }
    }

    Pixel pixel;
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