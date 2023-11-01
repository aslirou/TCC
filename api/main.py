import glob
from os import listdir

import cv2
import networkx as nx
import torch
from fastapi import FastAPI

from ..service.image_processing_service import analyze_and_graph_images

app = FastAPI()

state = {
    "home_return": {
        "message": "none",
        "folder": "TCC/images"
    },
    "model": None,
    "count": 0,
    "tag_dict": {}
}
DG = nx.DiGraph()


@app.on_event("startup")
async def startup_event():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True, device='cpu')
    state['model'] = model
    if state["home_return"]['message'] == "none":
        state["count"] += 1
        state["home_return"] = {
            'message': "hello vorldo!",
            'folder': '',
            'count': state["count"]
        }


@app.put('/')
async def update_folder(folder_path: str):
    folder_contents = listdir(folder_path)
    state["home_return"] = {
        'message': 'Loaded!',
        'folder': folder_path,
        'contents': folder_contents
    }
    return state["home_return"]


@app.get('/tags/')
async def create_graph():
    global model
    global home_return
    global tag_dict
    tag_dict = {}
    path = home_return["folder"]
    jpgs = glob.glob(path + '/**/*.jpg', recursive=True)
    pngs = glob.glob(path + '/**/*.png', recursive=True)
    jpegs = glob.glob(path + '/**/*.jpeg', recursive=True)
    webps = glob.glob(path + '/**/*.webp', recursive=True)

    images = jpgs + jpegs + pngs + webps
    images.reverse()
    for img in images[:150]:
        # img_name = img[len(img_folder):
        # global home_return].replace("\\", "/")

        result = model(img, size=1024)  # múltiplo de 32

        tags = result.pandas().xyxy[0][['name']].groupby("name").value_counts()
        ind = images.index(img)
        for j in range(len(tags)):
            tag_name = tags.index[j]
            tag_quant = tags.iloc[j]

            tag_img_json = {
                "image": str(img),
                "idents": int(tag_quant),
                "trust": "TBD"
            }
            if tag_name not in tag_dict.keys():
                tag_dict[tag_name] = []

            tag_dict[tag_name].append(tag_img_json)

            DG.add_edge(tag_name, img)  # ligando nome da classe identificada com o nome do arquivo
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


def prepare_img(image_path: str, transform):
    """
    Prepare your images here by loading and transforming into tensors
    """
    img_array = cv2.imread(image_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img_tensor = transform(img_array)  # Now use the transform function
    return img_tensor


@app.get("/analyze_images/")
def analyze_images(folder_path: str):
    graph = analyze_and_graph_images(folder_path)
    # Convert the graph structure into dictionary for jsonify it
    graph_dict = {
        "nodes": [{
            "id": node,
            **attr_dict
        } for node, attr_dict in graph.nodes(data=True)],
        "edges": [{
            "source": source,
            "target": target,
            **attr_dict
        } for source, target, attr_dict in graph.edges(data=True)]
    }
    return graph_dict
