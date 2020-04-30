nvcc main.cu
mkdir -p output/single

./a.out images/lena_rgb.png output/single/vertical_flip.png vflip single
./a.out images/lena_rgb.png output/single/horizontal_flip.png hflip single
./a.out images/lena_rgb.png output/single/blur.png blur single
./a.out images/lena_rgb.png output/single/sharpen.png sharpen single
./a.out images/lena_rgb.png output/single/edge.png edge single
./a.out images/lena_rgb.png output/single/grayscale.png gray single
./a.out images/lena_rgb.png output/single/weighted_grayscale.png grayweight single
