#pragma once
#include <stdint.h>
#ifdef __cplusplus 
extern "C"{
#endif

int32_t cuda_mean_filter(uint8_t *input, uint8_t *output,
                         uint32_t width, uint32_t height,
                         uint32_t window);

#ifdef __cplusplus 
}
#endif