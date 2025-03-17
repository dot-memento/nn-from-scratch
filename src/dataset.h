#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>

typedef struct dataset {
    size_t entry_count;
    size_t entry_size;
    size_t input_size;
    size_t output_size;
    double *data;
} dataset;

void dataset_split(const dataset *ds, dataset *training_ds, dataset *validation_ds, double split_ratio);
int dataset_load_csv(const char *filename, dataset *ds);

#endif // DATASET_H