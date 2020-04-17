#include <stdio.h>
#include <string.h>

#include "filters/blur_filter.h"
#include "consts.h"
#include "load_image.h"

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        printf("Incorrect number of arguments.\n");
        return 1;
    }

    const char* path_to_image = argv[1];
    const char* filter = argv[2];

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

    writeImage("out.png", image, width, height, channels);
    imageFree(image);
    
    return 0;
}