#include <stdio.h>
#include <string.h>

#include "image.h"
#include "filters/blur_filter.h"

const char* BLUR_FILTER = "blur";
const char* SHARPEN_FILTER = "sharpen";
const char* VERTICAL_FLIP_FILTER = "vflip";
const char* HORIZONTAL_FLIP_FILTER = "hflip";

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        printf("Incorrect number of arguments.\n");
        return 1;
    }

    const char* path_to_image = argv[1];
    const char* output_image = argv[2];
    const char* filter = argv[3];

    printf("Applying filter %s to image %s.\n", filter, path_to_image);

    int width, height, channels;
    stbi_uc* image = loadImage(path_to_image, &width, &height, &channels);
    
    if (image == NULL) {
        printf("Could not load image %s.\n", path_to_image);
        return 1;
    }

    Pixel pixel;
    getPixel(image, width, 5, 5, &pixel);
    printPixel(&pixel);

    if (strcmp(filter, BLUR_FILTER) == 0) {

    } else if (strcmp(filter, SHARPEN_FILTER) == 0) {
    
    } else if (strcmp(filter, VERTICAL_FLIP_FILTER) == 0) {

    } else if (strcmp(filter, HORIZONTAL_FLIP_FILTER) == 0) {

    } else {
        printf("Invalid filter %s.\n", filter);
    }

    writeImage(output_image, image, width, height, channels);
    imageFree(image);
    
    return 0;
}