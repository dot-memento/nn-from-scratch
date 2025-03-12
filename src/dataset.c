#include "dataset.h"

void dataset_split(const dataset *ds, dataset *training_ds, dataset *validation_ds, double split_ratio)
{
    *training_ds = (dataset) {
        .entry_count = ds->entry_count * split_ratio,
        .entry_size = ds->entry_size,
        .input_size = ds->input_size,
        .output_size = ds->output_size,
        .data = ds->data
    };

    *validation_ds = (dataset) {
        .entry_count = ds->entry_count - training_ds->entry_count,
        .entry_size = ds->entry_size,
        .input_size = ds->input_size,
        .output_size = ds->output_size,
        .data = ds->data + training_ds->entry_count * ds->entry_size
    };
}
