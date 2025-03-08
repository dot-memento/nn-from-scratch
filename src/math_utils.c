#include "math_utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "hyperparameters.h"

double binary_cross_entropy(const double predicted[], const double expected[], size_t size)
{
    double sum = 0;
    for (size_t i = 0; i < size; ++i)
        sum -= expected[i] * log(predicted[i] + FLT_MIN) + (1 - expected[i]) * log(1 - predicted[i] + FLT_MIN);
    return sum;
}

double rand_double()
{
    return (double)rand() / RAND_MAX;
}

double rand_double_in_range(float a, float b)
{
    return a + (b - a) * rand_double();
}

double sample_gaussian_distribution(double mu, double sigma)
{
    // Using Box-Muller transform
    double u1;
    do
        u1 = rand_double();
    while (u1 == 0);
    return mu + sigma * sqrt(-2 * log(u1)) * cos(2 * M_PI * rand_double());
}

static void memswap(void *a, void *b, size_t num)
{
    for (uint_fast8_t *p = a, *q = b, *sentry = p + num; p < sentry; ++p, ++q)
    {
        uint_fast8_t t = *p;
        *p = *q;
        *q = t;
    }
}

void shuffle(void *array, size_t count, size_t element_size)
{
    uint8_t *char_array = array;
    for (size_t i = count; i > 1; --i)
    {
        int j = rand() % i;
        memswap(char_array + element_size*(i-1), char_array + element_size*j, element_size);
    }
}
