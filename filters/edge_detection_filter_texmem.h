#ifndef EDGE_DETECTION_FILTER_TMEM_H
#define EDGE_DETECTION_FILTER_TMEM_H

#include "../image.h"
#include "util.h"

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texture_reference;
size_t pitch;

stbi_uc* edgeDetectionTexmem(stbi_uc* input_image, int width, int height, int channels);
__device__ bool isOutOfBounds(int x, int y, int image_width, int image_height);
__global__ void edgeDetectionTexmem(const stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int* mask, int mask_size, int shmem_width);
__global__ void test();

stbi_uc* edgeDetectionTexmem(stbi_uc* input_image, int width, int height, int channels) {
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


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    uchar4 *d_vector;
    cudaMalloc(&d_vector, channels * width *height * sizeof(uchar4));
    cudaBindTexture2D(NULL, texture_reference, d_vector, channelDesc, width, height, width * sizeof(uchar4));

    // cudaArray* cuArray;
    // cudaMallocArray(&cuArray, &channelDesc, width * channels, height);
    // cudaMemcpyToArray(cuArray, 0, 0, input_image, channels * width * height * sizeof(uchar4), cudaMemcpyHostToDevice);

    // cudaBindTextureToArray(texture_reference, cuArray, channelDesc);

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

    printf("Blocks %d, threads %d, total threads %d\n", blocks, threads, total_threads);
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    edgeDetectionTexmem<<<grid, block>>>(d_input_image, d_output_image_x, width, height, channels, total_threads, d_mask_x, mask_size, square_root);
    cudaDeviceSynchronize();
    edgeDetection<<<grid, block>>>(d_output_image_x, d_output_image_y, width, height, channels, total_threads, d_mask_y, mask_size, square_root);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&time, start, stop);
    printf("Texture memory: %f\n", time);

    cudaMemcpy(h_output_image, d_output_image_x, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void test() {
    const int x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_coordinate = blockIdx.y * blockDim.y + threadIdx.y;

    uchar4 test2 = tex2D(texture_reference, x_coordinate, y_coordinate);
    printf("got %u, %u, %u, %u", test2.x, test2.y, test2.z, test2.w);
}

__global__ void edgeDetectionTexmem(const stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int* mask, int mask_size, int shmem_width) {
    const int x_coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_coordinate = blockIdx.y * blockDim.y + threadIdx.y;
    
    int padding_size = mask_size / 2;

    int red = 0;
    int blue = 0;
    int green = 0;
    int alpha = 0;
    for (int i = 0; i < mask_size; i++) {
        for (int j = 0; j < mask_size; j++) {
            int current_x_global = x_coordinate - padding_size + i;
            int current_y_global = y_coordinate - padding_size + j;
            if (isOutOfBounds(current_x_global, current_y_global, width, height)) {
                continue;
            }
            uchar4 current_pixel = tex2D(texture_reference, x_coordinate, y_coordinate);
            int mask_element = mask[i * mask_size + j];

            red += current_pixel.x * mask_element;
            green += current_pixel.y * mask_element;
            blue += current_pixel.z * mask_element;
        }
    }

    Pixel pixel;
    // getPixel(input_image, padded_width, x_coordinate, y_coordinate, &pixel);
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