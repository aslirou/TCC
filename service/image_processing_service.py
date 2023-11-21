import os
import json
from random import random
import cv2
import torch
from PIL import Image
import networkx as nx
from torchvision.transforms import transforms
from yolov5.utils.general import check_requirements
from yolov5.utils.general import non_max_suppression
import matplotlib.pyplot as plt

check_requirements(('torch', 'PIL'))


def visualize_prediction(img_path, prediction, names):
    image = cv2.imread(img_path)
    for *box, prob, class_id in prediction:
        x1, y1, x2, y2 = map(int, box)
        class_label = names[int(class_id)]

        # Calculate the size of the text and adjust the box size and position accordingly
        text_scale = 0.6
        text_thickness = 2
        text = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]

        # Adjust box size
        y1 = max(y1 - text[1] - 5, 0)
        x2 = max(x2 + text[0] + 5, 0)

        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_label, (x1 + 5, y1 + text[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (36, 255, 12),
                    text_thickness)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.savefig(
        os.path.join('C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\processed_images', os.path.basename(img_path)))


def analyze_and_graph_images(folder_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Changed to yolov5x
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
        try:
            for idx, (*box, prob, cls) in enumerate(pred, start=1):
                class_label = names[int(cls.item())]
                unique_id = f'{img_name}_{class_label}_{idx}'
                node_id = f'{img_name}_{idx}'
                G.add_node(node_id, unique_id=unique_id, img_path=img_name,
                           class_label=class_label, prediction_probability=float(prob))
                visualize_prediction(os.path.join(folder_path, img_name), pred, names)
        except AttributeError:
            print(f'pred object: {pred} doesn\'t have xyxy attribute')

    for node1, data1 in G.nodes(data=True):
        for node2, data2 in G.nodes(data=True):
            if node1 != node2 and data1['class_label'] == data2['class_label']:
                avg_probability = (data1['prediction_probability'] + data2['prediction_probability']) / 2
                G.add_edge(node1, node2, weight=avg_probability)

    return G


def analyze_images(folder_path: str):
    g = analyze_and_graph_images(folder_path)
    graph_dict = {
        "nodes": [{
            "id": node,
            **attr_dict
        } for node, attr_dict in g.nodes(data=True)],
        "edges": [{
            "source": source,
            "target": target,
            "class_label": g.nodes[source]['class_label'],  # Use either source or target node's class_label.
            **attr_dict
        } for source, target, attr_dict in g.edges(data=True)]
    }
    with open('C:\\Users\\aslir\\Documents\\Faculdade\\TCC\\back\\json\\results_x.json', 'w') as outfile:
        json.dump(graph_dict, outfile)
    return graph_dict


def save_model(model, model_name="model.pth"):
    path = r"C:\Users\aslir\Documents\Faculdade\TCC\back\model"
    if not os.path.exists(path):
        os.makedirs(path)

    model_path = os.path.join(path, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
