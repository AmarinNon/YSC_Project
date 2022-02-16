from ttkbootstrap import Style
from tkinter import NO, ttk,messagebox
import runpy


# settings
style = Style(theme='superhero')
window = style.master
window.title("PESA v 0.0.1")

global Hz_model
global Distance_model
Hz_model = ["0"]
Distance_model = ["0"]
shoulder_model = ["0"]
# define function zone : change what the button are going to do here
def handcommand():
        print("hand!!")
        from handTrackerV2 import Hz
        if Hz is not None:
            Hz_model[0] = Hz

def shouldercommand():
        print("shoulder!!")
        from shoulderTrackerV2 import avg_value
        if avg_value is not None:
            shoulder_model[0] = avg_value


def gaitcommand():
        print("gait!!")
        from footTrackerV2 import status
        if status == "Normal":
            Distance_model[0] = "Normal"
        elif status == "Slow gait":
            Distance_model[0] = "Slow gait"
            
def Diagnosecommand():
    T = ""
    U = ""
    S = ""
    T_name = ""
    U_name = ""
    S_name = ""
    print(Hz_model[0])
    print(shoulder_model[0])
    print(Distance_model[0])
    if Hz_model[0] == "0" or shoulder_model[0] == "0" or Distance_model[0] == "0":
        print("Please check all symtoms")
        messagebox.showinfo(title="Diagnose", message="Please check all symtoms")
    if Hz_model[0] != "0" and shoulder_model[0] != "0" and Distance_model[0] != "0":
        
        if Hz_model[0] >= 2:
            print("[/] Tremor \n")
            T = "[/] Tremor \n"
            T_name = 1
        elif Hz_model[0] < 2:
            print("[X] Tremor \n")
            T = "[X] Tremor \n"
            T_name = 0
        if shoulder_model[0] >= 0.025:
            print("[/] Unsymetric \n")
            U = "[/] Unsymetric \n"
            U_name = 1
        elif shoulder_model[0] < 0.025:
            print("[X] Unsymetric \n")
            U = "[X] Unsymetric \n"
            U_name = 0
        if Distance_model[0] == "Slow gait":
            print("[/] Bradykinesia \n")
            S = "[/] Bradykinesia \n"
            S_name = 1
        elif Distance_model[0] == "Normal":
            print("[X] Bradykinesia \n")
            S = "[X] Bradykinesia \n"
            S_name = 0
        
        if T_name + U_name + S_name >= 2:
            messagebox.showinfo(title="Diagnose", message=T +  U + S + '\n' + 'Parkinson')
        elif T_name + U_name + S_name < 2:
            messagebox.showinfo(title="Diagnose", message=T +  U +  S + '\n' + 'Normal')
    

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

Diagnose = ttk.Button(
    window,
    text = "Diagnose",
    style = 'success.Outline.TButton',
    command = Diagnosecommand)
Diagnose.pack(side = 'left', padx = 5, pady = 10)

window.mainloop()