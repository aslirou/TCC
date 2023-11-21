import glob
import json
import time
from os import listdir
from typing import Optional

import networkx as nx
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..service.image_processing_service import analyze_images
from ..service.utils_service import check_and_update_images

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
image_json = "C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\back\\json\\image_list.json"
image_path = "C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\front\\image-analysis\\public\\images"
json_result = "C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\back\\json\\results_x.json"


@app.on_event("startup")
async def startup_event():
         start_time = time.time()
         analyze_images(image_path)
         end_time = time.time()  # Save the time at which the function execution has finished
         elapsed_time = end_time - start_time  # Calculate the time it took to run the function
         print(f"Time taken to run function 'analyze_images': {elapsed_time} seconds")


@app.get("/get_images_by_class/")
def get_images_by_class(class_label: Optional[str] = None):
    data = []
    if class_label:
        with open(json_result) as file:
            json_data = json.load(file)

            for node in json_data['nodes']:
                if node["class_label"] == class_label:
                    data.append(node)
    return data


@app.get("/get_all_images/")
def get_all_images():
    data = []
    with open(json_result) as file:
        json_data = json.load(file)

        for node in json_data['nodes']:
                data.append(node)
    return data

@app.get("/get_distinct_classes/")
def get_distinct_classes():
    class_set = set()  # a set data structure to hold unique classes

    # Open JSON file and read data
    with open(json_result, 'r') as file:
        json_data = json.load(file)

        # Loop over each node and add their class_label to the class_set
        for node in json_data['nodes']:
            class_set.add(node["class_label"])

    # Convert the set to a list and return it
    return list(class_set)


@app.get("/get_all_edges_by_weight/")
def get_images_by_class(weight: Optional[float] = None):
    data = []
    if weight:
        with open(json_result) as file:
            json_data = json.load(file)
            for node in json_data['edges']:
                if float(node["weight"]) >= weight:
                    data.append(node)
    return data
