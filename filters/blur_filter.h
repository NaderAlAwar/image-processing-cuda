#ifndef BLUR_FILTER_H
#define BLUR_FILTER_H

#include "../image.h"
#include "util.h"

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

    // Declare the 5x5 filtering kernel.
    const int kernelSize = 25;
    double myKernel[kernelSize];

    // Declare a "canvas" of pixels with the same dimensions as the filtering kernel.
    Pixel myCanvas[kernelSize];

    double scalability_factor; // Factor to scale the filtering kernel values.
    double center_coef;
    double adj_coef_inner;
    double corner_coef_inner;
    double adj_coef_outer;
    double adj_corner_coef_outer;
    double corner_coef_outer;

    bool gaussian = false;

    if(gaussian) { // Implement Gaussian Blur
        scalability_factor = 1.0f/273; // Factor to scale the filtering kernel values.
        center_coef = 41;
        adj_coef_inner = 26;
        corner_coef_inner = 16;
        adj_coef_outer = 7;
        adj_corner_coef_outer = 4;
        corner_coef_outer = 1;
    }

    else { // Implement Box Blur
        scalability_factor = 1.0f/kernelSize; // Factor to scale the filtering kernel values.
        center_coef = 1;
        adj_coef_inner = 1;
        corner_coef_inner = 1;
        adj_coef_outer = 1;
        adj_corner_coef_outer = 1;
        corner_coef_outer = 1;
    }

    center_coef *= scalability_factor;
    adj_coef_inner *= scalability_factor;
    corner_coef_inner *= scalability_factor;
    adj_coef_outer *= scalability_factor;
    adj_corner_coef_outer *= scalability_factor;
    corner_coef_outer *= scalability_factor;

    /* THE 5x5 FILTERING KERNEL
    *
    *   |   corner_coef_outer   | adj_corner_coef_outer | adj_coef_outer | adj_corner_coef_outer |   corner_coef_outer   |
    *   | adj_corner_coef_outer |   corner_coef_inner   | adj_coef_inner |   corner_coef_inner   | adj_corner_coef_outer |
    *   |     adj_coef_outer    |     adj_coef_inner    |   center_coef  |     adj_coef_inner    |     adj_coef_outer    |
    *   | adj_corner_coef_outer |   corner_coef_inner   | adj_coef_inner |   corner_coef_inner   | adj_corner_coef_outer |
    *   |   corner_coef_outer   | adj_corner_coef_outer | adj_coef_outer | adj_corner_coef_outer |   corner_coef_outer   |
    *
    */

    // Declare a value for filtering kernel weights to ignore (for cases when working with 
    // a pixel on the edge or in a corner of the image).
    double cancel_coef = 1000;

    // Fill the filtering kernel up with cancelling coefficients.
    for(int i = 0; i < kernelSize; i++) {
        myKernel[i] = cancel_coef;
    }

    for(int i = 0; i < kernelSize; i++) {
        // Iterate through each filtering kernel cell and assign weights given the position of the pixel on the image.
        switch(i) {
            
            // 1st row
            case 0:
                if(((x_coordinate - 2) >= 0) && ((y_coordinate - 2) >= 0)) {
                    myKernel[i] = corner_coef_outer;
                    getPixel(input_image, width, x_coordinate-2, y_coordinate-2, &myCanvas[i]);
                }
            break;

            case 1:
                if((x_coordinate - 1) >= 0 && (y_coordinate - 2) >= 0) {
                    myKernel[i] = adj_corner_coef_outer;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate-2, &myCanvas[i]);
                }
            break;

            case 2:
                if((y_coordinate - 2) >= 0) {
                    myKernel[i] = adj_coef_outer;
                    getPixel(input_image, width, x_coordinate, y_coordinate-2, &myCanvas[i]);
                }
            break;

            case 3:
                if((x_coordinate + 1) < width && (y_coordinate - 2) >= 0) {
                    myKernel[i] = adj_corner_coef_outer;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate-2, &myCanvas[i]);
                }
            break;

            case 4:
                if((x_coordinate + 2) < width && (y_coordinate - 2) >= 0) {
                    myKernel[i] = corner_coef_outer;
                    getPixel(input_image, width, x_coordinate+2, y_coordinate-2, &myCanvas[i]);
                }
            break;

            // 2nd row
            case 5:
                if((x_coordinate - 2) >= 0 && (y_coordinate - 1) >= 0) {
                    myKernel[i] = adj_corner_coef_outer;
                    getPixel(input_image, width, x_coordinate-2, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 6:
                if((x_coordinate - 1) >= 0 && (y_coordinate - 1) >= 0) {
                    myKernel[i] = corner_coef_inner;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 7:
                if((y_coordinate - 1) >= 0) {
                    myKernel[i] = adj_coef_inner;
                    getPixel(input_image, width, x_coordinate, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 8:
                if((x_coordinate + 1) < width && (y_coordinate - 1) >= 0) {
                    myKernel[i] = corner_coef_inner;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate-1, &myCanvas[i]);
                }
            break;

            case 9:
                if((x_coordinate + 2) < width && (y_coordinate - 1) >= 0) {
                    myKernel[i] = adj_corner_coef_outer;
                    getPixel(input_image, width, x_coordinate+2, y_coordinate-1, &myCanvas[i]);
                }
            break;

            // 3rd row
            case 10:
                if((x_coordinate - 2) >= 0) {
                    myKernel[i] = adj_coef_outer;
                    getPixel(input_image, width, x_coordinate-2, y_coordinate, &myCanvas[i]);
                }
            break;

            case 11:
                if((x_coordinate - 1) >= 0) {
                    myKernel[i] = adj_coef_inner;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate, &myCanvas[i]);
                }
            break;

            case 12:
                // No need to check, my pixel cell must be in the image.
                myKernel[i] = center_coef;
                getPixel(input_image, width, x_coordinate, y_coordinate, &myCanvas[i]);
            break;

            case 13:
                if((x_coordinate + 1) < width) {
                    myKernel[i] = adj_coef_inner;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate, &myCanvas[i]);
                }
            break;

            case 14:
                if((x_coordinate + 2) < width) {
                    myKernel[i] = adj_coef_outer;
                    getPixel(input_image, width, x_coordinate+2, y_coordinate, &myCanvas[i]);
                }
            break;

            // 4th row
            case 15:
                if((x_coordinate - 2) >= 0 && (y_coordinate + 1) < width) {
                    myKernel[i] = adj_corner_coef_outer;
                    getPixel(input_image, width, x_coordinate-2, y_coordinate+1, &myCanvas[i]);
                }
            break;

            case 16:
                if((x_coordinate - 1) >= 0 && (y_coordinate + 1) < width) {
                    myKernel[i] = corner_coef_inner;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate+1, &myCanvas[i]);
                }
            break;

            case 17:
                if((y_coordinate + 1) < width) {
                    myKernel[i] = adj_coef_inner;
                    getPixel(input_image, width, x_coordinate, y_coordinate+1, &myCanvas[i]);
                }
            break;

            case 18:
                if((x_coordinate + 1) < width && (y_coordinate + 1) < width) {
                    myKernel[i] = corner_coef_inner;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate+1, &myCanvas[i]);
                }
            break;

            case 19:
                if((x_coordinate + 2) < width && (y_coordinate + 1) < width) {
                    myKernel[i] = adj_corner_coef_outer;
                    getPixel(input_image, width, x_coordinate+2, y_coordinate+1, &myCanvas[i]);
                }
            break;

            // 5th row
            case 20:
                if((x_coordinate - 2) >= 0 && (y_coordinate + 2) < width) {
                    myKernel[i] = corner_coef_outer;
                    getPixel(input_image, width, x_coordinate-2, y_coordinate+2, &myCanvas[i]);
                }
            break;

            case 21:
                if((x_coordinate - 1) >= 0 && (y_coordinate + 2) < width) {
                    myKernel[i] = adj_corner_coef_outer;
                    getPixel(input_image, width, x_coordinate-1, y_coordinate+2, &myCanvas[i]);
                }
            break;

            case 22:
                if((y_coordinate + 2) < width) {
                    myKernel[i] = adj_coef_outer;
                    getPixel(input_image, width, x_coordinate, y_coordinate+2, &myCanvas[i]);
                }
            break;

            case 23:
                if((x_coordinate + 1) < width && (y_coordinate + 2) < width) {
                    myKernel[i] = adj_corner_coef_outer;
                    getPixel(input_image, width, x_coordinate+1, y_coordinate+2, &myCanvas[i]);
                }
            break;

            case 24:
                if((x_coordinate + 2) < width && (y_coordinate + 2) < width) {
                    myKernel[i] = corner_coef_outer;
                    getPixel(input_image, width, x_coordinate+2, y_coordinate+2, &myCanvas[i]);
                }
            break;

            default:
                printf("ERROR at first switch case in blur_filter.h.\n");
            break;
        }
    }

    for(int i = 0; i < avgPixelSize; i++) {
        // Iterate through all values of the pixel (R, G, B, and A).
        double weightedSum = 0;

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
                        // Alpha value (transparency).
                        weightedSum += myCanvas[j].a * myKernel[j];
                    break;

                    default:
                        printf("ERROR at second switch case in blur_filter.h.\n");
                    break;
                }
            }

        }

        // Copy the weighted sum of the specified value to the pixel's specified value.
        avgPixel[i] = (int)(weightedSum);

    }

    // Declare the output pixel
    Pixel outPixel;

    // POSSIBLY IMPLEMENT THE FOLLOWING CODE FOR EDGE DETECTION 
    // int sumAll = 0;
    // bool scaleDown = false;
    // for(int i = 0; i < avgPixelSize; i++) {
    //     if(avgPixel[i] > 255) {
    //         scaleDown = true;
    //         sumAll += avgPixel[i];
    //     }
    // }
    // if(!scaleDown) {
    //     // Fill in its value fields.
    //     outPixel.r = avgPixel[0];
    //     outPixel.g = avgPixel[1];
    //     outPixel.b = avgPixel[2];
    //     outPixel.a = avgPixel[3];
    // }
    // else {
    //     // Fill in its value fields.
    //     outPixel.r = avgPixel[0]/sumAll*255;
    //     outPixel.g = avgPixel[1]/sumAll*255;
    //     outPixel.b = avgPixel[2]/sumAll*255;
    //     outPixel.a = avgPixel[3]/sumAll*255;
    // }

    outPixel.r = avgPixel[0];
    outPixel.g = avgPixel[1];
    outPixel.b = avgPixel[2];
    outPixel.a = avgPixel[3];


    // OPTIONAL: Uncomment the following block and change x and y coordinates to print out the output pixels values
    //           of your choice.
    // if(x_coordinate == 0 && y_coordinate == 0) {
    //     printf("\n\nr = %d\n", outPixel.r);
    //     printf("g = %d\n", outPixel.g);
    //     printf("b = %d\n", outPixel.b);
    //     printf("a = %d\n", outPixel.a);
    // }

    // Set the pixel on the output image.
    setPixel(output_image, width, x_coordinate, y_coordinate, &outPixel);

}

#endif