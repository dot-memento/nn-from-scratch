#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>

typedef struct activation_pair {
    double (*base)(double x);
    double (*derivative)(double x);
} activation_pair;

extern const activation_pair activation_linear;
extern const activation_pair activation_sigmoid;
extern const activation_pair activation_tanh;
extern const activation_pair activation_relu;
extern const activation_pair activation_leaky_relu;
extern const activation_pair activation_swish;

#endif // ACTIVATION_H