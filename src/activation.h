#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>

typedef struct activation_pair {
    double (*base)(double x);
    double (*derivative)(double x);
} activation_pair;

double identity(double x);
double identity_derivative(double x);

double sigmoid(double x);
double sigmoid_derivative(double x);

double tanh_derivative(double x);

double relu(double x);
double relu_derivative(double x);

double leaky_relu(double x);
double leaky_relu_derivative(double x);

double swish(double x);
double swish_derivative(double x);

extern const activation_pair activation_linear;
extern const activation_pair activation_sigmoid;
extern const activation_pair activation_tanh;
extern const activation_pair activation_relu;
extern const activation_pair activation_leaky_relu;
extern const activation_pair activation_swish;

#endif // ACTIVATION_H