#include "stdio.h"
#include "stdlib.h"
#include "data_file.h"

int32_t read_data_file(const char *file_name, void *data_buf, uint32_t size)
{
    FILE *fp = NULL;
    fp = fopen(file_name, "rb");
    if (fp == NULL)
    {
        printf("Failed to open %s\n", file_name);
        return -1;
    }

    uint32_t read_cnt = fread(data_buf, 1, size, fp);
    if (read_cnt != size)
    {
        printf("read %d bytes, but need %d bytes\n", read_cnt, size);
        return -1;
    }
    return 0;
}

int32_t save_data_file(const char *file_name, void *data_buf, uint32_t size)
{
    FILE *fp = NULL;
    fp = fopen(file_name, "wb");
    if (fp == NULL)
    {
        printf("Failed to open %s\n", file_name);
        return -1;
    }

    uint32_t write_cnt = fwrite(data_buf, 1, size, fp);
    if (write_cnt != size)
    {
        printf("writed %d bytes, but need %d bytes\n", write_cnt, size);
        return -1;
    }
    return 0;
}

int32_t generate_rand_data(uint8_t *data_buf, uint32_t size)
{
    uint32_t i;
    for (i = 0; i < size; i++)
    {
        data_buf[i] = rand();
    }
    return 0;
}