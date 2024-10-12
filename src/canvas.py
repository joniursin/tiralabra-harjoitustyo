import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import train_network

class Canvas:
    def __init__(self, app):
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

        self.image = Image.new("L", (280, 280), color=0)
        self.image_draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(
            x - 10, y - 10, x + 10, y + 10, fill="black", width=10)
        self.image_draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill="white")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=0)
        self.image_draw = ImageDraw.Draw(self.image)
        self.predict_label.config(text="")

    def predict_image(self):
        resize_image = self.image.resize((28, 28))
        data = np.array(resize_image)/255.0
        data = np.array(data.reshape(-1, 1))
        predict = train_network.test_network(data)
        predict_number = np.argmax(predict)
        predict_percentage = predict[np.argmax(predict)]
        self.predict_label.config(text=f"Predicted '{predict_number}' ({float(predict_percentage)*100:,.2f}%)")

app = tk.Tk()
app_app = Canvas(app)
app.mainloop()
