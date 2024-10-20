import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import train_network

class Canvas:
    def __init__(self, app):
        """Creates canvas for drawing numbers

        Args:
            app (Tk): Tkinter gui window.
        """
        self.app = app
        self.app.title("Draw")
        self.canvas = tk.Canvas(app, width=280, height=280, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        self.clear_button = tk.Button(app, text="clear", command=self.clear)
        self.clear_button.pack(side=tk.RIGHT)

        self.predict_button = tk.Button(app, text="predict", command=self.predict_image)
        self.predict_button.pack(side=tk.LEFT)

        self.predict_label = tk.Label(app, text="")
        self.predict_label.pack()

        self.image = Image.new("L", (280, 280), color=255)
        self.image_draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        """Adds oval to the given canvas and ellipse to PIL image.

        Args:
            event (tk.Canvas): Canvas that is being updated.
        """
        x, y = event.x, event.y
        self.canvas.create_oval(
            x - 5, y - 5, x + 5, y + 5, fill="black", width=10)
        self.image_draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="black")
        self.predict_image()

    def clear(self):
        """Clears the canvas and PIL image.
        """
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.image_draw = ImageDraw.Draw(self.image)
        self.predict_label.config(text="")

    def predict_image(self):
        """Resizes the PIL image, turns it into np.array and sends it to the neural network for prediction.
        """
        resize_image = ImageOps.invert(self.image.resize((28, 28)))
        data = np.array(resize_image)/255.0
        data = np.array(data.reshape(-1, 1))
        predict = train_network.test_network(data)
        predict_number = np.argmax(predict)
        predict_percentage = predict[np.argmax(predict)]
        self.predict_label.config(text=f"Predicted '{predict_number}' ({float(predict_percentage)*100:,.2f}%)")

app = tk.Tk()
app_app = Canvas(app)
app.mainloop()
