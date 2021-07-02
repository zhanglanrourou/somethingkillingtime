#pragma once
#include "stdint.h"
#ifdef __cplusplus 
extern "C"{
#endif

int32_t read_data_file(const char *file_name, void *data_buf, uint32_t size);

int32_t save_data_file(const char *file_name, void *data_buf, uint32_t size);

int32_t generate_rand_data(uint8_t *data_buf, uint32_t size);

#ifdef __cplusplus 
}
#endif