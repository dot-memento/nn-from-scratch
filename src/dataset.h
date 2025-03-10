#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>

typedef struct dataset {
    size_t entry_count;
    size_t entry_size;
    size_t input_size;
    size_t output_size;
    double data[];
} dataset;

#endif // DATASET_H