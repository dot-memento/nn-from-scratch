#ifndef INITIALIZATION_H
#define INITIALIZATION_H

class Layer;

using initialization_function = void (*)(Layer *layer);

// Recommanded with tanh and sigmoid
void initialization_xavier(Layer *layer);

// Recommanded with ReLU and Swish
void initialization_he(Layer *layer);

#endif // INITIALIZATION_H