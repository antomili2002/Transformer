# Transformer Project

This project implements a Transformer model from scratch using PyTorch
Literature: "Attention is All You Need" by Vaswani et al.

## Project Structure

### 2.1 Creating a project directory

We will create a directory for the project. This directory will contain all the code for the project. We will call this directory `transformer_project`. You can create this directory in the location of your choice. For example, you can create it in your project directory.

### 2.2 How to structure the project

We use a modular approach, where we will create a separate python module for each component of the project. We will create the following modules (directories/scripts) for our project:

- `modelling`: This module will contain the code for the model architecture including the learning rate schedulers, loss functions and training code.
- `dataset.py`: This script will contain the code for loading, cleaning and preparing the data for model training.
- `test`: This directory will contain the code for testing the modules.
- `run/main.py`: This script will contain the code for running the model training and evaluation.
