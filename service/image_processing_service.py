import os
import json
import torch
from PIL import Image
import networkx as nx
from torchvision.transforms import transforms
from yolov5.utils.general import check_requirements
from yolov5.utils.general import non_max_suppression

check_requirements(('torch', 'PIL'))


def analyze_and_graph_images(folder_path, json_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)  # Changed to yolov5x
    model = model.to(device)
    names = model.module.names if hasattr(model, 'module') else model.names
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    G = nx.Graph()
    for img_name in os.listdir(folder_path):
        image = Image.open(os.path.join(folder_path, img_name))
        image_tensor = transform(image).unsqueeze(0).to(device)
        pred_raw = model(image_tensor)
        pred = non_max_suppression(pred_raw)[0]
        preds = []  # List to hold tuples of (class_label, prediction_probability)
        try:
            for *box, prob, cls in pred:
                if prob > 0.5:
                    class_label = names[int(cls.item())]
                    preds.append((img_name, {'class_label':class_label, 'prediction_probability':float(prob)}))
            # Add multiple nodes with attributes
            G.add_nodes_from(preds)
        except AttributeError:
            print(f'pred object: {pred} doesn\'t have xyxy attribute')
    for node1, data1 in G.nodes(data=True):
        for node2, data2 in G.nodes(data=True):
            if node1 != node2 and data1['class_label'] == data2['class_label']:
                avg_probability = (data1['prediction_probability'] + data2['prediction_probability']) / 2
                G.add_edge(node1, node2, weight=avg_probability)
    return G


def analyze_images(folder_path: str):
    graph = analyze_and_graph_images(folder_path,
                                     'C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\back\\json\\imagenet1000.json')
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
    with open('C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\back\\json\\results.json', 'w') as outfile:
        json.dump(graph_dict, outfile)
    return graph_dict