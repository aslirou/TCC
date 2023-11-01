import os

import networkx as nx
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models


def analyze_and_graph_images(folder_path):
    # Check if we have a CUDA-enabled GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a pre-trained ResNet model
    model = models.resnet50(pretrained=True)
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

    # Initialize an empty graph
    G = nx.Graph()

    # Classify each image and add it to a graph
    for img_name in os.listdir(folder_path):
        # Open image
        image = Image.open(os.path.join(folder_path, img_name))

        # Apply transformations
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get the model output
        output = model(image_tensor)
        softmax_output = F.softmax(output, dim=1)
        predicted_probs, predicted_classes = torch.max(softmax_output, 1)
        predict_prob = predicted_probs.item()
        predict_class = predicted_classes.item()

        # Add node with image name, class and probability as properties
        G.add_node(img_name, class_label=predict_class, prediction_probability=predict_prob)

    # Add edges if images have the same class label
    for node1, data1 in G.nodes(data=True):
        for node2, data2 in G.nodes(data=True):
            if node1 != node2 and data1['class_label'] == data2['class_label']:
                # For now, we set the edge weight as the average of two prediction probabilities
                G.add_edge(node1, node2, weight=(data1['prediction_probability'] + data2['prediction_probability']) / 2)

    return G
