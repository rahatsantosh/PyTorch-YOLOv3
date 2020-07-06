from models_yolo import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision import transforms

from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

tf1 = transforms.Compose(
    [transforms.Resize([416,416]),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.reshape(x, (1,3,416,416)))])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Darknet("../yolo_versions/yolov3.cfg", img_size=416).to(device)
try:
    model.load_darknet_weights("../yolo_versions/yolov3.weights")
    print("Done")
except:
    print("Not done")

classes = load_classes('./coco.names')

# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

def yolo_detect(img, model, classes, colors):
    img1 = Image.fromarray(img)

    with torch.no_grad():
                detections = model(tf1(img1))
                detections = non_max_suppression(detections, 0.8, 0.4)
    print(detections)

    if (detections is not None) and not all(x is None for x in detections):
        # Rescale boxes to original image
        detections = rescale_boxes(detections, 416, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            '''box_w = x2 - x1
            box_h = y2 - y1'''

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            op = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
            #bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            #ax.add_patch(bbox)
            # Add label
            cv2.putText(op,classes[int(cls_pred)],(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
            #img = cv2.putText(op, classes[int(cls_pred)], (x1, y1), color=(255,255,255), thickness=1, fontScale=1, font=cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.LINE_AA)
            '''plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )'''
    return np.array(img)


vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()
    if not ret:
    	continue
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img1 = yolo_detect(frame, model, classes, colors)

    cv2.imshow('YOLOv3_tiny', img1)

    # 'q' for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
