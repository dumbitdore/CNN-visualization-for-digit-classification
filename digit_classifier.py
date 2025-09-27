"""
Simple GUI in which a user can draw a digit and trigger digit recognition
"""

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os
import sys

import threading

try:
    from src.train import load_model, train_and_save
except Exception as e:
    print(f"error while importing from src.train: {e}")
    sys.exit(1)

MODEL_PATH = "models/cnn_model_full.h5"

class DigitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit classifier")

        # canvas with high resolution for synthesizing blurry lines
        self.canvas_size = 280 # image size * 10
        self.img_size = 28
        self.brush_size = 22 # ~ slightly less than 1 pixel of the final image

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=4, padx=8, pady=8)

        # PIL image as the ground truth (L for 8-bit grayscale)
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=255)
        self.im_draw = ImageDraw.Draw(self.image)

        # Mouse vent
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        # Buttons
        self.predict_button = tk.Button(root, text='Predict', command=self.predict, state='disabled')
        self.predict_button.grid(row=1, column=0, sticky='ew')

        tk.Button(root, text='Clear', command=self.clear).grid(row=1, column=1, sticky='ew')
        tk.Button(root, text='Quit', command=root.quit).grid(row=1, column=3, sticky='ew')

        self.train_button = tk.Button(root, text='(Re-)Train Model', command=self.train_model)
        self.train_button.grid(row=2, column=0, columnspan=4, sticky='ew')

        # although this probably will not be displayed
        self.status = tk.Label(root, text='Searching for model...')
        self.status.grid(row=3, column=0, columnspan=4, pady=6)

        self.model = None

        # check if a pretrained model is already available under the model path
        if os.path.exists(MODEL_PATH):
            # in this case a model was found
            try:
                self.model = load_model(MODEL_PATH)
                self.status.config(text='Model loaded.')
                self.predict_button.config(state='normal')
            except Exception as e:
                # the model couldnt be loaded
                self.status.config(text='Error loading the model. Please train the model first...')
        else:
            # no model found. asking the user to trigger the training
            self.status.config(text='Please train the model first...')

    def run_in_thread(self, target, *args):
        """
        Since actions like training will take several seconds to minutes depending on the available computing power,
        the ui would freeze for a longer period.
        As a solution, threading will be used.

        This method takes a target/ task and executes it independently
        """

        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()

    def paint(self, event):
        """
        This method pints a circle at the cursers position on click on the canvas as well as on the PIL image
        """

        x, y = event.x, event.y
        r = self.brush_size // 2
        # Draw on the visible tkinter canvas
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        # Draw on the PIL image (0 = black, 255 = white)
        self.im_draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear(self):
        """
        Clears canvas and resets PIL image
        """
        self.canvas.delete('all')
        self.im_draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)
        self.status.config(text='Canvas cleared.')

    def preprocess_image(self):
        """
        Preprocesses the image for training.

        The canvas has 10 times the size of the size the model expects.
        As the model is trained with MNIST data which has blurry lines only, the models performance is worse on images
        with black and white pixels only. Using a bigger canvas and downsizing later synthesizes those blurry lines quite well.
        """

        # Copy the drawn image
        img = self.image.copy()
        arr = np.array(img)
        # Invert if background is lighter than foreground
        if arr.mean() > 127:
            img = ImageOps.invert(img)
        # Resize to 28x28
        img = img.resize((self.img_size, self.img_size), resample=Image.LANCZOS)
        # Normalize to [0,1]
        x = np.array(img).astype('float32') / 255.0
        x = x.reshape(1, self.img_size, self.img_size, 1)
        return x

    def train_model(self):
        """
        Start the training with the loaded model in a second thread
        """

        # check if the model was loaded
        if self.model is not None:
            self.status.config(text='Training started...')
            self.predict_button.config(state='disabled')
            self.train_button.config(state='disabled')
            self.root.update_idletasks()

            def task():
                try:
                    train_and_save(save_path=MODEL_PATH)
                    model = load_model(MODEL_PATH)

                    self.root.after(0, lambda: self.on_model_loaded(model))
                except Exception as e:
                    self.root.after(0, lambda: self.status.config(text=f'Error: {e}'))

            self.run_in_thread(task)
        else:
            self.status.config(text='No Model found. Please train the model first!')

    def on_model_loaded(self, model):
        self.model = model
        self.predict_button.config(state='normal')
        self.train_button.config(state='normal')
        self.status.config(text='Training finished. Model loaded.')

    def predict(self):
        if self.model is None:
            self.status.config(text='No model loaded.')
            return

        x = self.preprocess_image()
        preds = self.model.predict(x)
        pred = int(np.argmax(preds, axis=1)[0])
        conf = float(np.max(preds))

        # Show top-3 predictions for debugging
        top3_idx = np.argsort(preds[0])[-3:][::-1]
        top3 = [(int(i), float(preds[0][i])) for i in top3_idx]

        self.status.config(text=f'Prediction: {pred}')

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()