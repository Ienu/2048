#!/usr/bin/python3

import tkinter
import tkinter.messagebox
import numpy as np
import random
import math
from PIL import Image, ImageTk
from core import Core


class GUI:
    def __init__(self):
        # core init
        self.core = Core()

        # GUI init
        self.root = tkinter.Tk()
        self.root.title('2048')
        self.root.geometry('600x800+10+10')
        # self.root.overrideredirect(True)

        # canvas init
        self.canvas_root = tkinter.Canvas(self.root, width=600, height=800)
        im_root = self._get_image('2048bkg.png', 600, 800)
        self.canvas_root.create_image(300, 400, image=im_root)
        self.canvas_root.place(x=0, y=0)

        # frame init
        self.frame = tkinter.Frame(self.root)
        self.frame.bind('<KeyRelease>', self._key_callback)

        self.frame.focus_set()
        self.frame.pack()

        # button init
        im_button = self._get_image('btn.png', 128, 39)
        self.btn = tkinter.Button(self.canvas_root, bd=0, image=im_button, width=128, height=39, command=self._new_game)
        self.btn.place(x=419, y=138)

        # last labels
        self.labels = []

        self.first_win = False

        # show
        self._show()
        self.root.mainloop()


    def _new_game(self):
        self.first_win = False
        self.core.reset()
        self._show()


    def _key_callback(self, event):
        if event.keysym == 'Up':
            if True == self.core.action_up():
                self.core.emerge()
                self._show()

        elif event.keysym == 'Left':
            if True == self.core.action_left():
                self.core.emerge()
                self._show()

        elif event.keysym == 'Right':
            if True == self.core.action_right():
                self.core.emerge()
                self._show()

        elif event.keysym == 'Down':
            if True == self.core.action_down():
                self.core.emerge()
                self._show()
        else:
            return False
        
        if True == self.core._test():
            tkinter.messagebox.showinfo(title='Game over!', message=self.msg[0])

        if self.core.suc2048 == True and self.first_win == False:
            tkinter.messagebox.showinfo(title='You win!', message=self.msg[1])
            self.first_win = True

        return True


    def _get_image(self, file_name, width, height):
        im = Image.open(file_name).resize((width, height))
        return ImageTk.PhotoImage(im)

    
    def _color(self, r, g, b): 
        color_str = '#%02x%02x%02x' % (int(r), int(g), int(b))
        return color_str


    def _block(self, value):
        # background color
        color_list = ['#000000', '#eee4da', '#ede0c8', '#f2b179', '#f59563', '#f67c5f', 
            '#f6623d', '#edcf72', '#eccb61', '#edc850', '#edc53f', '#edc22e']
        self.msg = ['\u6751\u957f\u662f\u4e2a\u5927\u7b28\u86cb\uff01',
            '\u6751\u957f~\u4f60\u597d\u5389\u5bb3\u8bf6\uff01']

        if value <= 2048:
            bg_color_str = color_list[int(math.log(value) / math.log(2))]
        else:
            bg_color_str = color_list[0]
        # 2:(238, 228, 218) # 4:(237, 224, 200) # 8:(242, 177, 121) # 16:(245, 149, 99) # 32:(246, 124, 95) # 64:(246, 98, 61) 
        # 128:(237, 207, 114) # 256:(236, 203, 97) # 512:(237, 200, 80) # 1024:(237, 197, 63) # 2048:(237, 194, 46)

        # front color
        fg_color_str = self._color(250, 248, 239)
        if value <= 4:
            fg_color_str = self._color(119, 110, 101)

        # font size
        font_size = 48
        if value > 100 and value < 1000:
            font_size = 36
        elif value > 1000 and value < 10000:
            font_size = 24
        elif value > 10000:
            font_size = 12

        im_block = self._get_image('block.png', 100, 100)
        label = tkinter.Label(self.canvas_root, image=im_block, bg=bg_color_str, width=100, height=100, 
            text=str(value), font='Helvetica -%d bold' % font_size, fg=fg_color_str, compound=tkinter.CENTER)

        return label


    def _show(self):
        for lab in self.labels:
            lab.destroy()

        for row in range(4):
            for col in range(4):
                value = self.core.board[row, col]
                if value != 0:
                    label = self._block(value)
                    label.place(x=64+col*121, y=237+row*121)
                    self.labels.append(label)



if __name__ == '__main__':
    gui = GUI()
    core = Core()
    