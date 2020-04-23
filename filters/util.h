#ifndef UTIL_H
#define UTIL_H

#include "../image.h"

const int MAX_THREADS = 1024;

stbi_uc* zeroPadImage(stbi_uc* input_image, int &width, int &height, int channels, int filterSize);

stbi_uc* zeroPadImage(stbi_uc* input_image, int &width, int &height, int channels, int filterSize) {
    int halfFilterSize = filterSize / 2;
    int paddedWidth = width + 2 * halfFilterSize;
    int paddedHeight = height + 2 * halfFilterSize;

    stbi_uc* paddedImage = (stbi_uc*) malloc(channels * paddedWidth * paddedHeight * sizeof(stbi_uc));

    Pixel zeroPixel = { .r = 0, .g = 0, .b = 0, .a = 0 };
    Pixel otherPixel;

    for (int i = 0; i < paddedWidth; i++) {
        for (int j = 0; j < paddedHeight; j++) {
            if (i < halfFilterSize || i > paddedWidth - halfFilterSize || j < halfFilterSize || j > paddedWidth - halfFilterSize) {
                setPixel(paddedImage, paddedWidth, i, j, &zeroPixel);
            } else {
                getPixel(input_image, width, i - halfFilterSize, j - halfFilterSize, &otherPixel);
                setPixel(paddedImage, paddedWidth, i, j, &otherPixel);
            }
        }
    }

    width = paddedWidth;
    height = paddedHeight;

    return paddedImage;
}

#endif