nvcc main.cu
mkdir -p images/batch
for i in {1..2000}; do cp images/lena_rgb.png "images/batch/$i.png"; done

mkdir -p expected_output/batch
./a.out images/batch expected_output/batch edge batch 