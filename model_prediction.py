import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt


IMAGES_FOLDER = "./images"
MODELS_FOLDER = "./model"
MODEL_NAME = "catvsdog.pt"
CLASSES = ["cat", "dog"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.jit.load(rf"{MODELS_FOLDER}/{MODEL_NAME}", map_location=torch.device('cpu'))
model.eval()

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((224, 224))
])

images = os.listdir(IMAGES_FOLDER)
images_list = []

for image in images:
    img_path = rf"{IMAGES_FOLDER}/{image}"

    img_arr = cv2.imread(img_path)
    pil_image = Image.fromarray(img_arr)
    custom_image_transformed = data_transforms(pil_image)

    with torch.no_grad():
        custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
        custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(device))
        class_index = custom_image_pred.round().int().item()

        cv2.putText(img_arr, f"Class: {CLASSES[class_index]}", (5, 25), cv2.FONT_HERSHEY_PLAIN, 2, (200, 20, 50), 2)

        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        images_list.append(img_arr)
        # cv2.imshow("res", img_arr)
        # cv2.waitKey(0)


fig = plt.figure(figsize=(10, 10))
for i in range(len(images_list)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images_list[i])
plt.show()
