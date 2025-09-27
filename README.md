# CNN Visualization for Digit Classification

This is a small side project to practice setting up Python projects in a clean and reusable way.
The CNN architecture and MNIST training were originally implemented during my dual studies. The current focus is on turning this into a standalone tool with a simple GUI and neuron activation visualizations.

A user can draw digits and see the model’s predictions. Future versions will also visualize neuron activations across the CNN layers. It has not yet been decided whether predictions and visualizations will run in real time or be triggered manually. The choice depends on the performance of both the prediction and the visualization.

## Current State

* A CNN model has been built with TensorFlow and trained on the MNIST dataset.
* Tests show excellent performance on the MNIST test data.
* A simple GUI for digit drawing has been implemented. Predictions on self-drawn digits are slightly less accurate than on the test dataset but still reliable.
* On startup, the tool checks if a trained model exists in a predefined path.
* Users can train a new model at any time. The trained model is then saved to the predefined path.
* After loading a model, users can draw a digit and trigger a prediction via a button. (Future versions may include real-time prediction.)

## Next Steps

* Implement a first MVP visualization of neuron activations.
* Define a concept for displaying convolutional layers and network behavior.

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/dumbitdore/CNN-visualization-for-digit-classification.git
cd CNN-visualization-for-digit-classification
```

Set up a Python environment and install dependencies. You can choose one of the following options:

### Option 1: Using Conda

```bash
conda create -n digit-cnn python=3.12
conda activate digit-cnn
conda install --file requirements.txt
```

### Option 2: Using pip

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Run the application:

```bash
python digit_classifier.py
```
