#include <stdio.h>
#include <string.h>

#include "image.h"
#include "filters/blur_filter.h"
#include "filters/sharpen_filter.h"
#include "filters/vertical_flip_filter.h"
#include "filters/horizontal_flip_filter.h"

const char* BLUR_FILTER = "blur";
const char* SHARPEN_FILTER = "sharpen";
const char* VERTICAL_FLIP_FILTER = "vflip";
const char* HORIZONTAL_FLIP_FILTER = "hflip";

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        printf("Incorrect number of arguments.\n");
        return 1;
    }

    const char* path_to_input_image = argv[1];
    const char* path_to_output_image = argv[2];
    const char* filter = argv[3];

    printf("Applying filter %s to image %s.\n", filter, path_to_input_image);

    int width, height, channels;
    stbi_uc* image = loadImage(path_to_input_image, &width, &height, &channels);
    
    if (image == NULL) {
        printf("Could not load image %s.\n", path_to_input_image);
        return 1;
    }

    stbi_uc* filtered_image;
    if (strcmp(filter, BLUR_FILTER) == 0) {
    } else if (strcmp(filter, SHARPEN_FILTER) == 0) {
        
    } else if (strcmp(filter, VERTICAL_FLIP_FILTER) == 0) {
        
    } else if (strcmp(filter, HORIZONTAL_FLIP_FILTER) == 0) {
        filtered_image = horizontalFlip(image, width, height, channels);
    } else {
        printf("Invalid filter %s.\n", filter);
    }

    writeImage(path_to_output_image, filtered_image, width, height, channels);
    imageFree(image);
    imageFree(filtered_image);
    
    return 0;
}