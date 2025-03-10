#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#ifndef M_PI
    #define M_PI 3.14159265358979323846264338327950288
#endif

#include <stddef.h>

double rand_double();
double rand_double_in_range(double a, double b);
double sample_gaussian_distribution(double mu, double sigma);

void shuffle(void *array, size_t count, size_t element_size);

#endif // MATH_UTILS_H