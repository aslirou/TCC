from fastapi import FastAPI
from tkinter import filedialog
import tkinter as tk
from os import listdir

import torch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

root = tk.Tk()
root.withdraw()

app = FastAPI()

count = 0
home_return = {
    "message"   : "none",
    "folder"    : "/run/media/rick/Extreme SSD/Personal/Albums/"
}

@app.get("/")
async def root():
    global home_return
    global count
    global model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True)

    if home_return['message'] == "none":
        count += 1
        home_return = {
            'message'   : "hello vorldo!",
            'folder'    : '',
            'count'     : count
        }

    global DG
    DG = nx.DiGraph()

    return home_return

@app.put('/')
async def update_folder(folder_path: str):
    global home_return

    folder_contents = listdir(folder_path)
    global image_folder
    image_folder = []

    home_return = {
        'message'   : 'Loaded!',
        'folder'    : folder_path,
        'contents'  : folder_contents
    }

    return home_return

@app.get('/tags/')
async def create_graph():
    global model
    global home_return
    global tag_dict
    tag_dict = {}
    path = home_return["folder"]
    jpgs = glob.glob( path + '/**/*.jpg', recursive=True)
    pngs = glob.glob( path + '/**/*.png', recursive=True)
    jpegs = glob.glob( path + '/**/*.jpeg', recursive=True)
    webps = glob.glob( path + '/**/*.webp', recursive=True)

    images = jpgs + jpegs + pngs + webps
    images.reverse()
    for img in images[:150]:
        # img_name = img[len(img_folder):
        # global home_return].replace("\\", "/")

        result = model(img, size=1024) # múltiplo de 32

        tags = result.pandas().xyxy[0][['name']].groupby("name").value_counts()
        ind = images.index(img)
        for j in range(len(tags)):
            tag_name = tags.index[j]
            tag_quant = tags.iloc[j]

            tag_img_json = {
                    "image" : str(img),
                    "idents": int(tag_quant),
                    "trust" : "TBD"
                }
            if tag_name not in tag_dict.keys():
                tag_dict[tag_name] = []

            tag_dict[tag_name].append(tag_img_json)

            DG.add_edge(tag_name, img) # ligando nome da classe identificada com o nome do arquivo
            if tag_name == 'person':
                if tag_quant == 2:
                    DG.add_edge('pair', img)
                elif tag_quant == 3:
                    DG.add_edge('trio', img)
                if tag_quant > 3:
                    DG.add_edge('group', img)
            if tag_name == 'cat' or tag_name == 'dog':
                DG.add_edge('pets', img)

        if len(tags) == 0:
            DG.add_node(img)
        print(f"{ind}/{len(images)} - processing!")

    return tag_dict
