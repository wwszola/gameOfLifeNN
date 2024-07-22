import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

import tensorflow as tf
import keras

from model import build_model

class SequencePlayer:
    def __init__(self, size):
        self.tape = []
        self.tape_pos = 0
        self.max_tape_size = 128
        self.size = size

        self.overcompleteness = 5
        self.build_model()
        self.random_init()

    def build_model(self):
        self.model = build_model(self.overcompleteness)
        self.model.build((1, *self.size, 1))

    def load_weights(self, path):
        self.model.load_weights(path)

    def random_init(self):
        arr = np.random.random(self.size)
        self.tape.append(arr)
    
    def prev_frame(self):
        self.tape_pos = max(0, self.tape_pos - 1)
        return self.tape[self.tape_pos]

    def next_frame(self):
        if self.tape_pos == len(self.tape) - 1:
            prev_frame = self.tape[self.tape_pos]
            frame = self.model(prev_frame.reshape(1, *self.size, 1))
            frame = frame.numpy().reshape(self.size)

            self.tape.append(frame)
            if len(self.tape) >= self.max_tape_size:
                self.tape.pop(0)
            else:
                self.tape_pos += 1
            
            return frame
        else:
            self.tape_pos = min(len(self.tape) - 1, self.tape_pos+1)
            return self.tape[self.tape_pos]

class View:
    def __init__(self, master, player):
        self.master = master
        self.player = player
        
        # Create widgets
        self.setup_canvas(400, 400)
        self.canvas.pack()

        self.setup_controls()
        self.controls_frame.pack(pady=10)

        self.setup_loading()
        self.loading_frame.pack(pady=10)


        self.show_array(self.player.prev_frame())

    def setup_canvas(self, width, height):
        self.canvas = tk.Canvas(self.master, width=width, height=height)
        self.img_tk = ImageTk.PhotoImage("L", (400, 400))
        self.image = self.canvas.create_image(
            0, 0, 
            anchor = tk.NW, 
            image = self.img_tk
        )

    def setup_controls(self):
        self.controls_frame = tk.Frame(self.master)
        
        left = tk.Button(self.controls_frame, text="←", command=self.prev_image)
        self.master.bind("<Left>", lambda _: self.prev_image())

        right = tk.Button(self.controls_frame, text="→", command=self.next_image)
        self.master.bind("<Right>", lambda _: self.next_image())

        randomize = tk.Button(self.controls_frame, text="Randomize", command=self.randomize)
        self.master.bind("r", lambda _: self.randomize())

        left.pack(side=tk.LEFT, padx=5)
        right.pack(side=tk.LEFT, padx=5)
        randomize.pack(side=tk.LEFT, padx=5)

    def setup_loading(self):
        self.loading_frame = tk.Frame(self.master)

        load_button = tk.Button(self.loading_frame, text="Load Weights", command=self.load_weights)
        load_button.pack(pady=5)

    def show_array(self, arr, vmin=0.0, vmax=1.0):
        arr = (arr - vmin)/(vmax - vmin) * 255.
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

        width, height = self.img_tk.width(), self.img_tk.height()

        img = Image.fromarray(arr)
        img = img.resize((width, height), resample=Image.Resampling.NEAREST)

        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.image, image = self.img_tk)

    def prev_image(self):
        self.show_array(self.player.prev_frame())
        
    def next_image(self):
        self.show_array(self.player.next_frame())

    def randomize(self):
        self.player.random_init()
        self.show_array(self.player.next_frame())

    def load_weights(self):
        filename = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if filename:
            print(filename)
            self.player.load_weights(filename)

if __name__ == "__main__":
    player = SequencePlayer((64, 64))
    root = tk.Tk()
    view = View(root, player)
    root.mainloop()
