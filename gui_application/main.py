# Program for determination of cardiovascular diseases by audio recording of the heartbeat

import neural_network_process
import tensorflow as tf
from tkinter import *
import pygame
from tkinter.ttk import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import tkinter.scrolledtext as tkst
import tkinter.messagebox as mb

classes = ['некорректная аудиодорожка', 'экстра-звуки', 'шумы в сердце', 'нормальное сердцебиение', 'экстрасистолия']
information = ['Данная аудиозапись не является \nзаписью биения сердца или \nсодержит значительные шумы',
               'Экстра-звуки могут появляться \nвремя от времени и могут быть \nидентифицированы из-за нарушения\nритма сердечного тона, \nсопровождающегося дополнительными или \nпропущенными сердечными сокращениями.\nЭто может быть признаком болезни.',
               'Шумы в сердце т.е. звуки сердца, \nпроизводимые, когда кровь \nперекачивается через сердечный \nклапан и создает звук, достаточно \nгромкий, чтобы его можно было \nуслышать с помощью стетоскопа.',
               'Фрагмент здорового сердцебиения',
               'Внеочередное преждевременное \nсердечное сокращение, \nдеполяризация и сокращение \nсердца или отдельных его камер,\nнаиболее часто регистрируемый\nвид аритмий.'
               ]

filename = ''
model = tf.keras.models.load_model('data/heart.h5')


# selection of an audio file for recognition
def open_file():
    global filename
    filename = askopenfilename(filetypes=[("Wave", "*.wav")])
    lbl['text'] = 'Открыт файл:\n' + filename
    btn1['state'] = 'normal'
    btn2['state'] = 'normal'
    text.delete('1.0', END)


# play the selected audio file
def play_audio():
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    pygame.event.wait()


def show_info():
    msg = "                    ВНИМАМАНИЕ!\nРезультат работы данного ПО не является медицинским заключением " \
          "\nДля подтверждения заключения необходимо обратиться к врачу "
    mb.showinfo("Информация", msg)


# output of the result of the neural network
def neural_network_output():
    if filename != '':
        answer = neural_network_process.sound_detection(filename, model)
        str = '==Результат распознавания: \n' + classes[answer] + "==\n"
        text.insert(INSERT, str + information[answer])
    else:
        text.insert(INSERT, 'Error')


if __name__ == "__main__":
    pygame.init()
    root = Tk()
    root.title("Распознавание аудиозаписи биения сердца")
    root.geometry('700x400')
    p1 = PhotoImage(file='data/icon.png')
    root.iconphoto(False, p1)
    opts = {'ipadx': 10, 'ipady': 10, 'padx': 10, 'pady': 10, 'sticky': 'nswe'}
    btn = ttk.Button(root, text='Открыть аудиозапись', command=lambda: open_file())
    btn.grid(column=1, row=1, **opts)

    lbl = Label(root, text="Откройте аудиофайл формата .wav")
    lbl.grid(column=1, row=6, rowspan=2, **opts)

    btn2 = ttk.Button(root, text='▶', width=40, command=lambda: play_audio())
    btn2['state'] = 'disabled'
    btn2.grid(column=1, row=2, **opts)

    btn1 = ttk.Button(root, text='Распознавание аудиофайла', command=lambda: neural_network_output())
    btn1['state'] = 'disabled'
    btn1.grid(column=1, row=3, **opts)

    info = ttk.Button(root, text='ИНФО', command=lambda: show_info())
    info.grid(column=4, row=0, ipadx=10, ipady=10)

    text = tkst.ScrolledText(root, width=30, height=10, font=("Times New Roman", 12))
    text.grid(column=2, row=1, columnspan=2, rowspan=8, **opts)

    mainloop()
