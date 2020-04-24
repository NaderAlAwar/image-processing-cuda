#ifndef BLUR_FILTER_H
#define BLUR_FILTER_H

#include "../image.h"

const int THREADS_PER_BLOCK = 512;

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
    // size_t limit;
    // cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    // cudaDeviceSetLimit(cudaLimitStackSize, limit * 50);
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

    const int avgPixelSize = 4;
    int avgPixel[avgPixelSize];

    const int kernelSize = 9;
    float myKernel[kernelSize];
    Pixel myCanvas[kernelSize];

    float corner_coef = 0.1111;
    float adj_coef = 0.1111;
    float center_coef = 0.1111;
    float cancel_coef = 1000;

    for(int i = 0; i < kernelSize; i++)
        myKernel[i] = cancel_coef;
    
    for(int i = 0; i < kernelSize; i++) {
        switch(i) {

            case 0:
                if(((x_coordinate - 1) >= 0) && ((y_coordinate - 1) >= 0)) {
                    myKernel[i] = corner_coef;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 1:
                if((y_coordinate - 1) >= 0) {
                    myKernel[i] = adj_coef;
                    getPixel(input_image, width, x_coordinate, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 2:
                if(((x_coordinate + 1) < width) && ((y_coordinate - 1) >= 0)) {
                    myKernel[i] = corner_coef;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 3:
                if((x_coordinate - 1) >= 0) {
                    myKernel[i] = adj_coef;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate, &myCanvas[i]);
                }
            break;

            case 4:
                // No need to check - my cell must be in the kernel.
                myKernel[i] = center_coef;
                getPixel(input_image, width, x_coordinate, y_coordinate, &myCanvas[i]);
            break;

            case 5:
                if((x_coordinate + 1) < width) {
                    myKernel[i] = adj_coef;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate, &myCanvas[i]);
                }
            break;

            case 6:
                if(((x_coordinate - 1) >= 0) && ((y_coordinate + 1) < height)) {
                    myKernel[i] = corner_coef;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate+1, &myCanvas[i]);
                }
            break;

            case 7:
                if((y_coordinate + 1) < height) {
                    myKernel[i] = adj_coef;
                    getPixel(input_image, width, x_coordinate, y_coordinate+1, &myCanvas[i]);
                }
            break;

            case 8:
                if(((x_coordinate + 1) < width) && ((y_coordinate + 1) < height)) {
                    myKernel[i] = corner_coef;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate+1, &myCanvas[i]);
                }
            break;

            default:
                printf("ERROR at first switch case in blur_filter.h.\n");
            break;

        }
    }

    for(int i = 0; i < avgPixelSize; i++) {

        float sum = 0;

        for(int j = 0; j < kernelSize; j++) {
            if(myKernel[j] != cancel_coef) {
                switch(i) {
                    case 0:
                        sum += myCanvas[j].r * myKernel[j];
                    break;

                    case 1:
                        sum += myCanvas[j].g * myKernel[j];
                    break;

                    case 2:
                        sum += myCanvas[j].b * myKernel[j];
                    break;

                    case 3:
                        sum += myCanvas[j].a * myKernel[j];
                    break;

                    default:
                        printf("ERROR at second switch case in blur_filter.h.\n");
                    break;
                }
            }

            else {
                myCanvas[j].r = 0;
                myCanvas[j].g = 0;
                myCanvas[j].b = 0;
                myCanvas[j].a = 0;
            }

        }

        avgPixel[i] = (int)sum;


    }

    Pixel outPixel;

    outPixel.r = avgPixel[0];
    outPixel.g = avgPixel[1];
    outPixel.b = avgPixel[2];
    outPixel.a = avgPixel[3];

    if(x_coordinate == 511 && y_coordinate == 0) {
        printf("\n\nr = %d\n", outPixel.r);
        printf("g = %d\n", outPixel.g);
        printf("b = %d\n", outPixel.b);
        printf("a = %d\n", outPixel.a);
    }


    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel);

}

#endif