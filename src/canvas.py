import tkinter as tk
from PIL import Image
import numpy as np
import train_network
import os


class Canvas:
    def __init__(self, app):
        self.app = app
        self.app.title("Draw")
        self.canvas = tk.Canvas(app, width=280, height=280, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        self.clear_button = tk.Button(app, text="clear", command=self.clear)
        self.clear_button.pack(side=tk.RIGHT)

        self.image = Image.new("L", (28, 28), color=0)

    def draw(self, event):
        x, y = event.x, event.y
        if 0 <= x < 280 and 0 <= y < 280:
            self.canvas.create_oval(
                x, y, x + 10, y + 10, fill="black", width=2)
            self.image.putpixel((x // 10, y // 10), 255)

            self.predict_image()

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)

    def predict_image(self):
        data = np.array(self.image)/255
        data = np.array(data.reshape(-1, 1))
        predict = train_network.test_network(data)
        predict_number = np.argmax(predict)
        predict_percentage = predict[np.argmax(predict)]

        print(
            f"Predicted '{predict_number}' {float(predict_percentage)*100:,.2f}%")


os.system("train_network.py")
app = tk.Tk()
app_app = Canvas(app)
app.mainloop()
