#pragma once
#include "stdio.h"

#ifdef __cplusplus 
extern "C"{
#endif

#define RETURN_IF_FAILED(ret) do{ \
    if (ret != 0)                 \
    {                             \
        return ret;               \
    }                             \
}while (0)                        \

#define RETURN_IF_NULL(ptr) do{       \
    if (ptr == NULL)                  \
    {                                 \
        printf("%s is NULL\n", #ptr); \
        return -1;                    \
    }                                 \
}while (0)                            \

#ifdef __cplusplus 
}
#endif