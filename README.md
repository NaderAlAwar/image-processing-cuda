# image-processing-cuda

This repository contains the codebase to run various parallel GPU based algorithms for image processing. Some of the algorithms implemented are image blurring, image flipping, and more. These parallel algorithms are run on a GPU using CUDA.

Note: You must have the ability to run CUDA files on your end in order to render any of the work in this repository. For more information about CUDA, please visit this link: https://developer.nvidia.com/about-cuda

Before any filters can be applied, the `main.cu` file must be compiled. To do that, open your terminal and run the following command from the root directory of this project:

```nvcc main.cu```

You can ignore any warnings that are printed to the console. A file named `a.out` should now be stored in the root directory.

To apply a filter to an image, please follow the next steps:
* Import an image of your choice in the `images` directory, or just use one of the images already there.
* From the root directory, run `a.out` with the following arguments (see filter arguments in the table):
```./a.out path_to_image_input path_to_image_output filter_name```
* Conversely, you can run the following command:
```sbatch runs/filter_run``` (please check the `runs` directory to see which file you should call).

### Table 1: Filters and their arguments
|      Filter     |  Filter Arg |
|:---------------:|:-----------:|
| Horizontal Flip | hflip       |
| Vertical Flip   | vflip       |
| Sharpening      | sharpen     |
| Blurring        | blur        |

