nvcc main.cu
unzip images/batch.zip -d images/batch

mkdir -p expected_output/batch
./a.out images/batch expected_output/batch edge batch 