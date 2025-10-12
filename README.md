# CNN Visualization for Digit Classification

This is a small side project to practice setting up Python projects in a clean and reusable way.
The CNN architecture and MNIST training were originally implemented during my dual studies. The current focus is on turning this into a standalone tool with a simple GUI and neuron activation visualizations.

## Features

1. **Digit Classification**: Users can draw digits in the main GUI, and the CNN predicts their label.
2. **Neuron Activation Visualization**: A separate visualization window can be triggered from the main GUI. This window shows:
   - Each layer of the CNN as a column of neurons or filters.
   - Activations of neurons using a blue colormap.
   - The predicted output neuron highlighted in red.
   - Connections between neurons, optionally colored based on activation values.
3. **Model Management**:
   - Checks for an existing trained model at startup.
   - Allows users to train and save a new model.

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
