#ifndef BLUR_FILTER_H
#define BLUR_FILTER_H

#include "../image.h"

const int THREADS_PER_BLOCK = 256;

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

    // Declare coordinates based on thread_id and image dimensions.
    int x_coordinate = thread_id % height;
    int y_coordinate = thread_id / width;

    // Declare an array of size 4 for values R, G, B, and A of the new pixel.
    const int avgPixelSize = 4;
    int avgPixel[avgPixelSize];

    // Declare the 3x3 filtering kernel.
    const int kernelSize = 9;
    float myKernel[kernelSize];

    // Declare a "canvas" of pixels with the same dimensions as the filtering kernel.
    Pixel myCanvas[kernelSize];

    // Assign the weights of each filtering kernel value.
    float center_coef = 0.25;
    float adj_coef = 0.125;
    float corner_coef = 0.0625;

    // Declare a value for filtering kernel weights to ignore (for cases when working with 
    // a pixel on the edge or in a corner of the image).
    float cancel_coef = 1000;

    // Fill the filtering kernel up with cancelling coefficients.
    for(int i = 0; i < kernelSize; i++)
        myKernel[i] = cancel_coef;
    
    for(int i = 0; i < kernelSize; i++) {
        // Iterate through each filtering kernel cell and assign weights given the position of the pixel on the image.
        switch(i) {

            case 0:
                if(((x_coordinate - 1) >= 0) && ((y_coordinate - 1) >= 0)) {
                    // If the top left kernel cell is NOT out of the bounds of the image.
                    myKernel[i] = corner_coef;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 1:
                if((y_coordinate - 1) >= 0) {
                    // If the top center kernel cell is NOT out of the bounds of the image.
                    myKernel[i] = adj_coef;
                    getPixel(input_image, width, x_coordinate, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 2:
                if(((x_coordinate + 1) < width) && ((y_coordinate - 1) >= 0)) {
                    // If the top right kernel cell is NOT out of the bounds of the image.
                    myKernel[i] = corner_coef;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 3:
                if((x_coordinate - 1) >= 0) {
                    // If the center left kernel cell is NOT out of the bounds of the image.
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
                    // If the center right kernel cell is NOT out of the bounds of the image.
                    myKernel[i] = adj_coef;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate, &myCanvas[i]);
                }
            break;

            case 6:
                if(((x_coordinate - 1) >= 0) && ((y_coordinate + 1) < height)) {
                    // If the bottom left kernel cell is NOT out of the bounds of the image.
                    myKernel[i] = corner_coef;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate+1, &myCanvas[i]);
                }
            break;

            case 7:
                if((y_coordinate + 1) < height) {
                    // If the bottom center kernel cell is NOT out of the bounds of the image.
                    myKernel[i] = adj_coef;
                    getPixel(input_image, width, x_coordinate, y_coordinate+1, &myCanvas[i]);
                }
            break;

            case 8:
                if(((x_coordinate + 1) < width) && ((y_coordinate + 1) < height)) {
                    // If the bottom right kernel cell is NOT out of the bounds of the image.
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
        // Iterate through all values of the pixel (R, G, B, and A).
        float weightedSum = 0;

        for(int j = 0; j < kernelSize; j++) {
            // Iterate through each filtering kernel cell
            if(myKernel[j] != cancel_coef) {
                // If the filtering kernel cell is inside of the picture, multiply its weight by the pixel's value.
                switch(i) {
                    case 0:
                        // Red value.
                        weightedSum += myCanvas[j].r * myKernel[j];
                    break;

                    case 1:
                        // Green value.
                        weightedSum += myCanvas[j].g * myKernel[j];
                    break;

                    case 2:
                        // Blue value.
                        weightedSum += myCanvas[j].b * myKernel[j];
                    break;

                    case 3:
                        // Alpha value (opaqueness).
                        weightedSum += myCanvas[j].a * myKernel[j];
                    break;

                    default:
                        printf("ERROR at second switch case in blur_filter.h.\n");
                    break;
                }
            }

            else {
                // If the filtering kernel cell is NOT inside of the picture, assign zero.
                myCanvas[j].r = 0;
                myCanvas[j].g = 0;
                myCanvas[j].b = 0;
                myCanvas[j].a = 0;
            }

        }

        // Copy the weighted sum of the specified value to the pixel's specified value.
        avgPixel[i] = (int)weightedSum;


    }

    // Declare the output pixel
    Pixel outPixel;

    // Fill in its value fields.
    outPixel.r = avgPixel[0];
    outPixel.g = avgPixel[1];
    outPixel.b = avgPixel[2];
    outPixel.a = avgPixel[3];
    // outPixel.a = myCanvas[4].a;

    // OPTIONAL: Uncomment the following block and change x and y coordinates to print out the output pixels values
    //           of your choice.
    // if(x_coordinate == 511 && y_coordinate == 0) {
    //     printf("\n\nr = %d\n", outPixel.r);
    //     printf("g = %d\n", outPixel.g);
    //     printf("b = %d\n", outPixel.b);
    //     printf("a = %d\n", outPixel.a);
    // }

    // Set the pixel on the output image.
    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel);

}

#endif