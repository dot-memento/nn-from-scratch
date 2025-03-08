#ifndef INITIALIZATION_H
#define INITIALIZATION_H

typedef struct layer layer;

typedef void (*initialization_function)(layer *layer);

// Recommanded with tanh and sigmoid
void initialization_xavier(layer *layer);

// Recommanded with ReLU and Swish
void initialization_he(layer *layer);

#endif // INITIALIZATION_H