#include "dataset.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include "constants.h"

#define CSV_CELL_BUFFER_SIZE 4096

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

int dataset_load_csv(const char *filename, dataset *ds)
{
    FILE *file = fopen(filename, "r");

    if (!file)
    {
        fprintf(stderr, PROGRAM_NAME": error: can't open '%s': %s\n", filename, strerror(errno));
        return 1;
    }

    size_t entry_size = SIZE_MAX;
    size_t current_entry_size = 1;
    size_t entry_count = 0;
    bool empty = true;

    int c = fgetc(file);
    while (c != EOF)
    {
        empty = false;

        if (c == ',')
        {
            current_entry_size++;
        }

        if (c == '\n')
        {
            if (entry_size == SIZE_MAX)
                entry_size = current_entry_size;
            else if (entry_size != current_entry_size)
            {
                fprintf(stderr, PROGRAM_NAME": error: entry %zu in '%s' doesn't have %zu fields\n", entry_count, filename, entry_size);
                return 1;
            }
            entry_count++;
            current_entry_size = 1;
            empty = true;
        }

        c = fgetc(file);
    }

    if (!empty)
    {
        entry_count++;
        if (entry_size == SIZE_MAX)
            entry_size = current_entry_size;
        else if (current_entry_size != 1 && entry_size != current_entry_size)
        {
            fprintf(stderr, PROGRAM_NAME": error: entry %zu in '%s' doesn't have %zu fields\n", entry_count, filename, entry_size);
            return 1;
        }
    }
    else if(entry_count == 0)
    {
        fprintf(stderr, PROGRAM_NAME": error: file '%s' doesn't have any entry\n", filename);
        return 1;
    }

    ds->entry_count = entry_count;
    ds->entry_size = entry_size;


    fseek(file, 0, SEEK_SET);

    double *data = malloc(entry_count * entry_size * sizeof(double));

    size_t offset = 0;
    char str_buffer[CSV_CELL_BUFFER_SIZE];
    c = 0;
    while (c != EOF)
    {
        size_t i = 0;
        while (i < CSV_CELL_BUFFER_SIZE-1)
        {
            c = fgetc(file);
            if (c == EOF || c == ',' || c == '\n')
                break;

            str_buffer[i++] = (char)c;
        }
        str_buffer[i++] = '\0';
        
        char *final_char;
        double number = strtod(str_buffer, &final_char);
        if (*final_char != '\0')
        {
            fprintf(stderr, PROGRAM_NAME": error: can't convert '%s' to a number\n", str_buffer);
            return 1;
        }
        
        data[offset++] = number;
    }

    ds->data = data;

    fclose(file);

    return 0;
}
