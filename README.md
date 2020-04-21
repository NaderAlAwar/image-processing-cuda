# image-processing-cuda

This repository contains the codebase to run various parallel GPU based algorithms for image processing. Some of the algorithms implemented are image blurring, image flipping, and more. These parallel algorithms will be run on a GPU using CUDA.

Before any filters can be applied, the `main.cu` file must be compiled. To do that, run the following command on your terminal from the root directory of this project:

```nvcc main.cu```

To apply any filter to the image of your

