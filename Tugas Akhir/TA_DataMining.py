import matplotlib.pyplot as plt
import csv
import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def prediksi_lulus(total_hadir):
    batas_lulus = 10
    return "ya" if total_hadir >= batas_lulus else "tidak"


data = []
with open(r'D:\Udinus\SMT 4\Data Mining\Tugas\Data_TA.csv', newline='') as file:
    reader = csv.reader(file, delimiter=';')
    next(reader)
    data = list(reader)

siswa_hadir = {}
for row in data:
    nm_siswa = row[1]
    if nm_siswa in siswa_hadir:
        siswa_hadir[nm_siswa] += row[3:16].count('Hadir')
    else:
        siswa_hadir[nm_siswa] = row[3:16].count('Hadir')


for row in data:
    nm_siswa = row[1]
    total_hadir = siswa_hadir[nm_siswa]
    status_lulus = prediksi_lulus(total_hadir)
    row.append(status_lulus)


root = tk.Tk()
root.title("Data Absensi")
tree = ttk.Treeview(root)
tree["columns"] = ("Nama Siswa", "Total", "Lulus")

tree.heading("#0", text="No")
tree.heading("#1", text="Nama Siswa")
tree.heading("#2", text="Total Hadir")
tree.heading("#3", text="Lulus")

tree.column("#0", width=50)
tree.column("#1", width=150)
tree.column("#2", width=100)
tree.column("#3", width=100)

for i, row in enumerate(data):
    no = row[0]
    nm_siswa = row[1]
    total_hadir = siswa_hadir[nm_siswa]
    lulus = row[18]

    tree.insert(parent='', index='end', id=i, text=no,
                values=(nm_siswa, total_hadir, lulus))

tree.pack()

total_lulus = sum(1 for row in data if row[18] == 'ya')
total_tdk_lulus = sum(1 for row in data if row[18] == 'tidak')

label_total_lulus = tk.Label(
    root, text=f"Total Lulus: {total_lulus} Siswa")
label_total_lulus.pack()

label_total_tdk_lulus = tk.Label(
    root, text=f"Total tidak Lulus: {total_tdk_lulus} Siswa")
label_total_tdk_lulus.pack()

X = [row[3:16] for row in data]
y = [row[18] for row in data]


label_encoder = LabelEncoder()
X_encoded = []
for row in X:
    encoded_row = []
    for val in row:
        encoded_val = label_encoder.fit_transform([val])[0]
        encoded_row.append(encoded_val)
    X_encoded.append(encoded_row)


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=1)


clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


accuracy = metrics.accuracy_score(y_test, y_pred)

label_accuracy = tk.Label(
    root, text=f"Akurasi: {accuracy*100} %")
label_accuracy.pack()


def create_tree():
    root = tk.Tk()
    root.title("Tree GUI")

    label_jdl = tk.Label(
        root, text=f"TREE")
    label_jdl.pack()

    tree_canvas = tk.Canvas(root, width=400, height=400)
    tree_canvas.pack()

    tree_canvas.create_line(200, 200, 200, 300)

    tree_canvas.create_line(200, 300, 150, 350)
    tree_canvas.create_line(200, 300, 250, 350)

    tree_canvas.create_rectangle(195, 185, 205, 195, fill='blue')
    tree_canvas.create_rectangle(190, 180, 210, 200, fill='blue')

    tree_canvas.create_oval(145, 345, 155, 355, fill='green')
    tree_canvas.create_oval(140, 340, 160, 360, fill='green')

    tree_canvas.create_oval(245, 345, 255, 355, fill='red')
    tree_canvas.create_oval(240, 340, 260, 360, fill='red')

    tree_canvas = tk.Canvas(root, width=400, height=400)
    tree_canvas.pack()

    tree_canvas.create_rectangle(105, 85, 115, 95, fill='blue')
    tree_canvas.create_rectangle(100, 80, 120, 100, fill='blue')
    tree_canvas.create_text(150, 90, text="Start",
                            fill="black", font=("Arial", 10))

    tree_canvas.create_oval(105, 125, 115, 135, fill='green')
    tree_canvas.create_oval(100, 120, 120, 140, fill='green')
    tree_canvas.create_text(
        230, 130, text="Total Hadir >= 10, Siswa Lulus", fill="black", font=("Arial", 10))

    tree_canvas.create_oval(105, 165, 115, 175, fill='red')
    tree_canvas.create_oval(100, 160, 120, 180, fill='red')
    tree_canvas.create_text(
        240, 170, text="Total Hadir < 10, Siswa Tidak Lulus", fill="black", font=("Arial", 10))


create_tree()

root.mainloop()
