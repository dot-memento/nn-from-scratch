# Neural Networks from Scratch in C 

This repository contains a lightweight neural network implementation in C. It provides a framework for creating, training, and evaluating neural networks with configurable architectures through JSON configuration.

## Features

- Neural network implementation in pure C
- JSON-based configuration for network architecture
- AdamW optimizer
- CSV output for loss and accuracy tracking and results visualization

## Usage

Use `make` to build from source and run the executable. The JSON configuration file can be provided as an argument, otherwise the file `./config.json` will be used.

## Limitations

- Many hardcoded things, including the input format, output format and metrics
- Minimal error handling
- Only supports feed-forward network
- CPU only

## Licence

[MIT License](LICENSE)
