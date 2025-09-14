import os

from src.train import train_and_save, load_model
from src.test_model import run_test

MODEL_PATH = "models/cnn_model_full.h5"

def main():
    # check if weights already exist
    if not os.path.exists(MODEL_PATH):
        print("No weights found. Starting training...")
        train_and_save(save_path=MODEL_PATH)
        print("Training done")
    else:
        print("Saved pretrained model found")

    try:
        print("Loading model...")
        model = load_model(MODEL_PATH)
        print("Loading done")
    except FileNotFoundError:
        print("Problem while loading the model. Exiting")
        return

    print("Testing the model with MNIST test data and visualizing confusion matrix...")
    run_test(model)

if __name__ == "__main__":
    main()