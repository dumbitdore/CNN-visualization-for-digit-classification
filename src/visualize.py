import tkinter as tk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm

MODEL_PATH = "models/cnn_model_full.keras"


class VisualizeWindow:
    """
    creates window for the visualization of the CNN and its activations
    colors neurons based on activation and highlights the predicted output neuron
    """

    def __init__(self, parent_app, line_color_by_activation=True):
        """
        Behavior:
        Initializes a new top-level Tkinter window and sets up ui components

        Lifecycle:
        The window exists as long as the user keeps it open.
        It can be closed using the 'Close' button, destroying the top-level window

        Parameters:
        Parent_app (object): parent application, which must contain the model and preprocessing method
        Line_color_by_activation (bool, optional): if True, lines connecting neurons are colored based on target neuron activations
                                                   default = true

        Returns:None
        """

        self.parent_app = parent_app

        # create new window attached to parent app
        self.top = tk.Toplevel(parent_app.root)
        self.top.title("CNN Visualization Enhanced")

        self.line_color_by_activation = line_color_by_activation

        tk.Label(self.top, text="CNN Visualization", font=("Arial", 12, "bold")).pack(padx=6, pady=6)

        # frame for buttons
        button_frame = tk.Frame(self.top)
        button_frame.pack(fill="x", padx=6, pady=6)

        # buttons
        tk.Button(button_frame, text="Draw Network", command=self.draw_network).pack(side="left", expand=True, fill="x", padx=3)
        tk.Button(button_frame, text="Close", command=self.top.destroy).pack(side="left", expand=True, fill="x", padx=3)

        # placeholders for matplotlib figure and canvas
        self.fig = None
        self.canvas = None

    def draw_network(self) -> None:
        """
        Visualizes the CNN

        Sequentially draws each neural network layer as a column of circles (neurons/filters), colored by their activation.
        Above each layer, a text label shows its name and size.
        Between each pair of layers, draws lines (connections) colored by the next-layer neurons activation if enabled
        for visual appeal. Although this is not technically accurate since the connections themselves do not have activations.
        The output neuron corresponding to the models prediction is highlighted in red.

        Behavior:
        1. Preprocesses the input image using the parents preprocessing method.
        2. Creates a new "activation model" that outputs all intermediate layer activations (Conv2D and Dense layers).
        3. Runs the activation model and the original model to get activations and predictions.
        4. Iteratively visualizes each layer:
            Conv2D: uses spatially averaged activations for each filter.
            Dense: uses the raw activation vector.
        5. Colors neurons using matplotlibs Blues colormap.
        6. Highlights the predicted output neuron in red.
        7. Draws connections between layers, optionally colored by target neuron activation.

        Lifecycle:
        Can be called multiple times to update the visualization for different input images.
        Old matplotlib canvas is destroyed before creating a new one.

        Returns: None
        """

        # load model from parent app
        model = self.parent_app.model
        if model is None:
            print("No model loaded.")
            return

        # retrieve and preprocess image via parent apps preprocess_image()
        x = self.parent_app.preprocess_image()

        # select all Conv2D and Dense layers
        target_layers = [layer for layer in model.layers if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))]

        # create "activation" model that outputs all intermediate activations
        activation_model = tf.keras.Model(inputs=model.input,outputs=[layer.output for layer in target_layers])

        # run prediction on activation model for activations
        activations = activation_model.predict(x)

        # run original model for the "real" prediction
        prediction = model.predict(x)
        prediction_class = int(np.argmax(prediction))

        # create matplotlib figure
        fig, ax = plt.subplots(figsize=(28, 12))
        ax.axis("off")
        ax.set_aspect('equal', 'box') # ensure circles are round and not squished

        # horizontal spacing between layers
        x_spacing = 12
        layer_x_positions = np.arange(len(target_layers)) * x_spacing

        # determine maximum number of units for vertical centering
        max_units = 0
        for act in activations:
            if len(act.shape) == 4:
                n_units = act.shape[-1]
            else:
                n_units = act.shape[-1]
            max_units = max(max_units, n_units)

        # vertical spacing between neurons
        y_spacing = 0.8

        # placeholders for previous activations and positions. None for first iteration
        prev_positions = None
        prev_activations = None
        neuron_radius = 0.2

        # iterate over layers
        for i, (layer, act) in enumerate(zip(target_layers, activations)):
            layer_x = layer_x_positions[i]
            current_activations = []

            # check whether current layer is Conv2D or Dense
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Conv2D Layers
                total_filters = act.shape[-1]
                vals = act.mean(axis=(0, 1, 2))

                # calculate vertical positions
                y_positions = np.linspace(
                    start=(total_filters-1)/2*y_spacing,
                    stop=-(total_filters-1)/2*y_spacing,
                    num=total_filters
                )

                # create viz for neurons
                for j, val in enumerate(vals):
                    color = cm.Blues(val)
                    circ = plt.Circle((layer_x, y_positions[j]), neuron_radius, color=color, ec="black", lw=0.5)
                    ax.add_patch(circ)
                    current_activations.append(val)

                # draw layer name and number of filters
                ax.text(layer_x, max(y_positions)+y_spacing*0.5,
                        f"{layer.name}\n{total_filters} filters",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

            elif isinstance(layer, tf.keras.layers.Dense):
                # Dense Layer
                vec = act[0]
                n_units = len(vec)
                is_output = (i == len(target_layers) - 1) # check if final layer

                # calculate vertical positions
                y_positions = np.linspace(
                    start=(n_units-1)/2*y_spacing,
                    stop=-(n_units-1)/2*y_spacing,
                    num=n_units
                )

                for j, val in enumerate(vec):
                    color = cm.Blues(val)
                    circ = plt.Circle((layer_x, y_positions[j]), neuron_radius, color=color, ec="black", lw=0.5)

                    if is_output and j == prediction_class:
                        # if final layer and predicted neuron: highlight neuron red
                        circ.set_edgecolor("red")
                        circ.set_linewidth(2.0)
                    ax.add_patch(circ)
                    if is_output:
                        # if final layer: draw the predicted label (digit)
                        ax.text(layer_x + 0.25, y_positions[j], str(j), va="center", fontsize=10)
                    current_activations.append(val)

                # draw layer name and number of filters
                ax.text(layer_x, max(y_positions)+y_spacing*0.5,
                        f"{layer.name}\n{n_units} neurons",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

            # draw connections
            if prev_positions is not None: # only if not first layer as a previous layer is needed
                # draw lines from previous layers neuron to the current neuron

                curr_vals = np.array(current_activations)

                # normalize activations for color mapping [0, 1]
                if curr_vals.max() > curr_vals.min(): # only if not all activations are the same
                    norm_vals = (curr_vals - curr_vals.min()) / (curr_vals.max() - curr_vals.min())
                else:
                    # if all activations are the same, use zeroes to avoid division by zero
                    norm_vals = np.zeros_like(curr_vals)

                # for each pair of previous and current neuron: draw a line
                for y_prev, act_prev in zip(prev_positions, prev_activations):
                    for y_curr, norm_act in zip(y_positions, norm_vals):

                        # depending on line_color_by_activation color by activation level else color gray
                        color = cm.Blues(norm_act) if self.line_color_by_activation else "gray"

                        # draw line
                        ax.plot([layer_x - x_spacing, layer_x],
                                [y_prev, y_curr],
                                color=color,
                                linewidth=0.3,
                                alpha=0.6)

            # save current positions and activation as previous for next iteration
            prev_positions = y_positions
            prev_activations = current_activations

        fig.tight_layout()

        # destroy if already exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # display
        self.fig = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.top)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(padx=5, pady=5, fill="both", expand=True)
