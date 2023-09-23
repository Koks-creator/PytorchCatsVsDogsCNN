import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGES_FOLDER = "./images"
MODELS_FOLDER = "./model"
MODEL_NAME = "catvsdog_resnet.pt"
CLASSES = ["dog", "cat"]
MEAN = np.array([0.5, 0.5, 0.5])
STD = np.array([0.25, 0.25, 0.25])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.jit.load(rf"{MODELS_FOLDER}/{MODEL_NAME}", map_location=device)
model.eval()

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.Resize((224, 224))
])

images = os.listdir(IMAGES_FOLDER)
images_list = []

for image in images:
    img_path = rf"{IMAGES_FOLDER}/{image}"

    img_arr = cv2.imread(img_path)
    pil_image = Image.fromarray(img_arr)
    custom_image_transformed = data_transforms(pil_image)
    print(custom_image_transformed.shape)
    with torch.no_grad():
        prediction_raw = model(custom_image_transformed.unsqueeze(dim=0).to(device))
        prediction = torch.sigmoid(prediction_raw)
        class_index = torch.round(prediction).int().item()

        cv2.putText(img_arr, f"Class: {CLASSES[class_index]}", (5, 50), cv2.FONT_HERSHEY_PLAIN, 3, (200, 20, 50), 2)

        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        images_list.append(img_arr)
        # cv2.imshow("res", img_arr)
        # cv2.waitKey(0)


fig = plt.figure(figsize=(10, 10))
for i in range(len(images_list)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images_list[i])
plt.show()
