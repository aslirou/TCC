import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import networkx as nx
import timm


def analyze_and_graph_images(folder_path, json_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    model = model.to(device)
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    with open(json_path, 'r') as j:
        id_to_labels = json.load(j)

    G = nx.Graph()

    for img_name in os.listdir(folder_path):
        image = Image.open(os.path.join(folder_path, img_name))

        image_tensor = transform(image).unsqueeze(0).to(device)

        output = model(image_tensor)
        softmax_output = F.softmax(output, dim=1)
        predicted_probs, predicted_classes = torch.max(softmax_output, 1)
        predict_prob = predicted_probs.item()
        predict_class = predicted_classes.item()

        if predict_prob < 0.1:
            continue

        label_names = id_to_labels[str(predict_class)]
        primary_label_name = label_names.split(",")[0]

        G.add_node(img_name, class_label=primary_label_name, prediction_probability=predict_prob)

    for node1, data1 in G.nodes(data=True):
        for node2, data2 in G.nodes(data=True):
            if node1 != node2 and data1['class_label'] == data2['class_label']:
                avg_probability = (data1['prediction_probability'] + data2['prediction_probability']) / 2
                G.add_edge(node1, node2, weight=avg_probability)

    return G
