#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "stdint.h"
#include "meanfilter_test.h"
#include "data_file.h"
#include "common.h"
#include "opencv2/opencv.hpp"
#include "cu_mean_filter.h"
#include "data_path.h"

#define INPUT_DATA_PATH "meanfilter/input.bin"
#define GT_DATA_PATH "meanfilter/gt_output.bin"
#define OUT_DATA_PATH "meanfilter/output.bin"
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define WINDOW 5

void mean_filter(uint8_t *input, uint8_t *output,
                 uint32_t width, uint32_t height,
                 uint32_t window)
{
    uint32_t row, col, idx;
    uint32_t window_radis = window >> 1;
    for (row = 0; row < height; row++)
    {
        if ((row < window_radis) || (row >= height - window_radis))
        {
            memcpy(&output[row * width], &input[row * width], width);
            continue;
        }

        for (col = 0; col < width; col++)
        {
            uint32_t sum = 0, i, j;
            idx = row * width + col;
            if ((col < window_radis) || (col >= width - window_radis))
            {
                output[idx] = input[idx];
                continue;
            }
            
            uint8_t *ptr_filter_adder = &input[idx - window_radis * width - window_radis];
            for (i = 0; i < window; i++)
            {
                for (j = 0; j < window; j++)
                {
                    uint8_t adder = ptr_filter_adder[i * width + j];
                    sum += *ptr_filter_adder;
                }
            }
            output[idx] = sum / (window * window);
        }
    }
}

using namespace cv;
int32_t meanfilter_test()
{
    int32_t ret;
    uint8_t *image = (uint8_t *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT);
    RETURN_IF_NULL(image);

    ret = read_data_file(INPUT_DATA_PATH, image, IMAGE_WIDTH * IMAGE_HEIGHT);
    RETURN_IF_FAILED(ret);

    uint8_t *output_image = (uint8_t *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT);
    RETURN_IF_NULL(output_image);

    mean_filter(image, output_image, IMAGE_WIDTH, IMAGE_HEIGHT, WINDOW);

    ret = save_data_file(GT_DATA_PATH, output_image, IMAGE_WIDTH * IMAGE_HEIGHT);
    RETURN_IF_FAILED(ret);

    uint8_t *gt_image = (uint8_t *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT);
    RETURN_IF_NULL(gt_image);

    cuda_mean_filter(image, output_image, IMAGE_WIDTH, IMAGE_HEIGHT, WINDOW);

    ret = save_data_file(OUT_DATA_PATH, output_image, IMAGE_WIDTH * IMAGE_HEIGHT);
    RETURN_IF_FAILED(ret);
    return 0;
}