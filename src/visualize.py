import tkinter as tk
import numpy as np
import sys

import tensorflow as tf

try:
    from src.train import load_model, train_and_save
except Exception as e:
    print(f"error while importing from src.train: {e}")
    sys.exit(1)

MODEL_PATH = "models/cnn_model_full.keras"

class VisualizeWindow:
    """
    defines a standalone window for visualization of the CNN
    at the moment only prints activations in console
    """

    def __init__(self, parent_app):
        # initialize window

        self.parent_app = parent_app
        self.top = tk.Toplevel(parent_app.root)
        self.top.title("Visualizer")

        tk.Label(self.top, text="Visualizer (Console)").grid(row=0, column=0, columnspan=2, padx=6, pady=6)

        # Buttons
        tk.Button(self.top, text='Load activations', command=self.load_and_print_activations).grid(row=1, column=0, sticky='ew', padx=6, pady=6)
        tk.Button(self.top, text='Close', command=self.top.destroy).grid(row=1, column=1, sticky='ew', padx=6, pady=6)

    def load_and_print_activations(self):
        """
        loads the activations of
        """

        # load the model from the parent app as its already loaded
        model = self.parent_app.model
        if model is None:
            print("Visualizer: No model found. Please train the model first.")
            return

        # use preprocessing method from parent app (loading current image drawn in parent app)
        # the image is loaded on button click so that its actually the latest version of the image even if drawn after the start of the viz window
        x = self.parent_app.preprocess_image()  # shape: (1, 28, 28, 1)

        # choose layers (conv2d and dense only)
        target_layers = [layer for layer in model.layers if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))]

        # check if layers could be extracted
        if not target_layers:
            # console outputs only as its a test version. will later be substituted by a label in the app
            print("Could not find any layer in the model")
            return

        # Create copy of the model, which delivers the targeted layers
        activation_model = tf.keras.Model(inputs=model.input, outputs=[layer.output for layer in target_layers])

        print("Input shape:", x.shape)

        activations = activation_model.predict(x)

        for i, (layer, act) in enumerate(zip(target_layers, activations)):
            print(f"\nLayer {i}: name={layer.name} type={layer.__class__.__name__}")
            print("shape:", act.shape)

            arr = np.array(act)
            print("  dtype:", arr.dtype)
            print("  min/max/mean/std:", float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std()))

            # conv-output
            if arr.ndim == 4:
                batch, H, W, C = arr.shape
                per_ch_mean = arr.mean(axis=(0, 1, 2))
                nonzero_count = np.count_nonzero(arr)
                total = arr.size
                print(f"  conv: HxW={H}x{W}, channels={C}, nonzero={nonzero_count}/{total} ({nonzero_count/total:.3f})")

                n_show = min(10, C)
                print("  per-channel mean (first %d): %s" % (n_show, np.round(per_ch_mean[:n_show], 4).tolist()))

                top_k = 5
                top_idx = np.argsort(per_ch_mean)[::-1][:top_k]
                print("  top channels (by mean):", [(int(i), float(np.round(per_ch_mean[i], 4))) for i in top_idx])

            # Dense-Output: (batch, units)
            elif arr.ndim == 2:
                vec = arr[0]
                n = vec.shape[0]
                n_show = min(20, n)
                print(f"  dense: units={n}")
                print("  first values:", np.round(vec[:n_show], 5).tolist())
                if n > n_show:
                    print("  ...")
            else:
                print("  unexpected activation ndim:", arr.ndim)

        print("=== Visualizer: Activations end ===\n")