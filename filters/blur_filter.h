#ifndef BLUR_FILTER_H
#define BLUR_FILTER_H

#include "../image.h"

const int THREADS_PER_BLOCK = 1024;

stbi_uc* blur(stbi_uc* input_image, int width, int height, int channels);
__global__ void blurKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads);

stbi_uc* blur(stbi_uc* input_image, int width, int height, int channels) {
    int image_size = channels * width * height * sizeof(stbi_uc);
    stbi_uc* d_input_image;
    stbi_uc* d_output_image;
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);

    cudaMallocManaged(&d_input_image, image_size);
    cudaMallocManaged(&d_output_image, image_size);
    cudaMemcpy(d_input_image, input_image, image_size, cudaMemcpyHostToDevice);

    int total_threads = width * height;
    int threads = min(THREADS_PER_BLOCK, total_threads);
    int blocks = (threads == total_threads) ? 1 : total_threads / THREADS_PER_BLOCK;

    printf("Blocks %d, threads %d\n", blocks, threads);
    size_t limit;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, limit * 50);
    blurKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, total_threads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;
}

__global__ void blurKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads) {
    
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (thread_id >= total_threads) {
        return;
    }

    int x_coordinate = thread_id % height;
    int y_coordinate = thread_id / width;

    int avgPixelSize = 4;
    int* avgPixel = (int*)malloc(avgPixelSize*sizeof(int));
 
    if(x_coordinate == 0) {
        if(y_coordinate == 0) {

            int kernelSize = 4;

            Pixel* myKernel = (Pixel*)malloc(kernelSize*sizeof(Pixel));

            getPixel(input_image, width, x_coordinate, y_coordinate, &myKernel[0]);
            getPixel(input_image, width, x_coordinate+1, y_coordinate, &myKernel[1]);
            getPixel(input_image, width, x_coordinate, y_coordinate+1, &myKernel[2]);
            getPixel(input_image, width, x_coordinate+1, y_coordinate+1, &myKernel[3]);

            for(int i = 0; i < avgPixelSize; i++) {
                int sum = 0;
                for(int j = 0; j < kernelSize; j++) {
                    switch(i){
                        case 0: sum += myKernel[j].r;
                                break;
                        case 1: sum += myKernel[j].g;
                                break;
                        case 2: sum += myKernel[j].b;
                                break;
                        case 3: sum += myKernel[j].a;
                                break;
                        default: printf("Error assigning sum");
                                break;
                    }
                }
                avgPixel[i] = sum/kernelSize;
            }

            printf("red = %d\n", avgPixel[0]);
            printf("green = %d\n", avgPixel[1]);
            printf("blue = %d\n", avgPixel[2]);
            printf("a = %d\n", avgPixel[3]);

        }
    }

    Pixel outPixel;

    outPixel.r = (stbi_uc)avgPixel[0];
    outPixel.g = (stbi_uc)avgPixel[1];
    outPixel.b = (stbi_uc)avgPixel[2];
    outPixel.a = (stbi_uc)avgPixel[3];

    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel);

}

#endif