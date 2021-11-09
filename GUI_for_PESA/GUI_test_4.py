from ttkbootstrap import Style
from tkinter import ttk
import runpy
# settings
style = Style(theme='superhero')
window = style.master
window.title("PESA v 0.0.1")

# define function zone : change what the button are going to do here
def handcommand():
        print("hand!!")
        runpy.run_module(mod_name='handTracker copy')

def shouldercommand():
        print("shoulder!!")
        runpy.run_module(mod_name='shoulderTracker')

def gaitcommand():
        print("gait!!")
        runpy.run_module(mod_name='footTracker')

# text zone
label = ttk.Label(
    text = 'PESA',
    font = ("Helvetica", 35))
label.pack(ipadx = 0, ipady = 0)

label = ttk.Label(
    text = 'A Parkinson\'s disease screening system',
    font = ("Helvetica", 15))
label.pack(ipadx = 0, ipady = 0)

label = ttk.Label(
    text = '\n This is the demo version ~ All rights reserved',
    font = ("Helvetica", 7))
label.pack(ipadx = 0, ipady = 0)

label = ttk.Label(
    text = 'Gifted Computer Plus, The Prince Royal\'s College',
    font = ("Helvetica", 7))
label.pack(ipadx = 0, ipady = 0)

# button zone
hand = ttk.Button(
    window,
    text = "Hand Detection",
    style = 'success.Outline.TButton',
    command = handcommand)
hand.pack(side = 'left', padx = 5, pady = 10)

shoulder = ttk.Button(
    window,
    text = "Shoulder Detection",
    style = 'success.Outline.TButton',
    command = shouldercommand)
shoulder.pack(side = 'left', padx = 5, pady = 10)

gait = ttk.Button(
    window,
    text = "Gait Detection",
    style = 'success.Outline.TButton',
    command = gaitcommand)
gait.pack(side = 'left', padx = 5, pady = 10)

window.mainloop()