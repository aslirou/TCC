import glob
import json
from os import listdir
from typing import Optional

import networkx as nx
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..service.image_processing_service import analyze_images

app = FastAPI()
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    print("Analisando Imagens")
    analyze_images("C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\front\\image-analysis\\public\\images")
    print("Imagens Analisadas!")


@app.get("/get_images_by_class/")
def get_images_by_class(class_label: Optional[str] = None):
    data = []
    if class_label:
        with open('C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\back\\json\\results.json', 'r') as file:
            json_data = json.load(file)

            for node in json_data['nodes']:
                if node["class_label"] == class_label:
                    data.append(node)
    return data


@app.get("/get_distinct_classes/")
def get_distinct_classes():
    class_set = set()  # a set data structure to hold unique classes

    # Open JSON file and read data
    with open('C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\back\\json\\results.json', 'r') as file:
        json_data = json.load(file)

        # Loop over each node and add their class_label to the class_set
        for node in json_data['nodes']:
            class_set.add(node["class_label"])

    # Convert the set to a list and return it
    return list(class_set)
