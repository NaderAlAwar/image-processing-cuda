#include <stdio.h>
#include <string.h>

#include "image.h"
#include "filters/blur_filter.h"
#include "filters/sharpen_filter.h"
#include "filters/vertical_flip_filter.h"
#include "filters/horizontal_flip_filter.h"
#include "filters/grayscale_filter.h"
#include "filters/grayscale_weighted_filter.h"
#include "filters/edge_detection_filter.h"
#include "filters/edge_detection_filter_shmem.h"
#include "filters/edge_detection_filter_cmem.h"
#include "filters/edge_detection_filter_texmem.h"

const char* BLUR_FILTER = "blur";
const char* SHARPEN_FILTER = "sharpen";
const char* VERTICAL_FLIP_FILTER = "vflip";
const char* HORIZONTAL_FLIP_FILTER = "hflip";
const char* GRAYSCALE_FILTER = "gray";
const char* GRAYSCALE_WEIGHTED_FILTER = "grayweight";
const char* EDGE_DETECTION_FILTER = "edge";

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
        filtered_image = blur(image, width, height, channels);
    } else if (strcmp(filter, SHARPEN_FILTER) == 0) {
        filtered_image = sharpen(image, width, height, channels);
    } else if (strcmp(filter, VERTICAL_FLIP_FILTER) == 0) {
        filtered_image = verticalFlip(image, width, height, channels);
    } else if (strcmp(filter, HORIZONTAL_FLIP_FILTER) == 0) {
        filtered_image = horizontalFlip(image, width, height, channels);
    } else if (strcmp(filter, GRAYSCALE_FILTER) == 0) {
        filtered_image = gray(image, width, height, channels);
    } else if (strcmp(filter, GRAYSCALE_WEIGHTED_FILTER) == 0) {
        filtered_image = grayWeight(image, width, height, channels);
    } else if (strcmp(filter, EDGE_DETECTION_FILTER) == 0) {
        filtered_image = edgeDetection(image, width, height, channels);
        filtered_image = edgeDetectionCmem(image, width, height, channels);
        filtered_image = edgeDetectionShmem(image, width, height, channels);
        filtered_image = edgeDetectionTexmem(image, width, height, channels);
    }  else {
        printf("Invalid filter %s.\n", filter);
    }

    writeImage(path_to_output_image, filtered_image, width, height, channels);
    imageFree(image);
    imageFree(filtered_image);
    
    return 0;
}