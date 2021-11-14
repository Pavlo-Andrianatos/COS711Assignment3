# Imports
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

import copy

import cv2

import torch
import torchvision.transforms
from PIL.ImagePath import Image
from torch.utils.data.dataset import T_co

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch.nn.functional as Functional

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import warnings


# Custom dataset for pytorch
class passion_fruit_dataset(Dataset):
    # Initialisation function, takes in a data frame and a directory for the input images
    def __init__(self, data_frame, image_directory):
        super().__init__()

        self.data_frame = data_frame
        self.image_id = self.data_frame['Image_ID'].unique()
        self.image_directory = image_directory  # Path(image_directory)

    # __getitem__ is called by the pytorch data loader when feeding data into the models for training or inference
    def __getitem__(self, index) -> T_co:
        image_id = self.image_id[index]

        # Returns record from data frame where the image id is equal to the one saved in item
        records = self.data_frame[self.data_frame['Image_ID'] == image_id]

        # Get .jpg file from image directory and convert to RGB
        image_name = image_id + '.jpg'
        image = Image.open(self.image_directory + image_name).convert("RGB")
        image = transforms.ToTensor()(image)

        # Constructs the bounding boxes for pytorch to use, is in the coco format
        boxes = records[['xmin', 'ymin', 'width', 'height']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Class labels for item
        labels = torch.tensor(records['class'].values, dtype=torch.int64)

        # Targets property, is a dictionary which contains the bounding boxes, class, and image id for item
        targets = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index])}

        # Returns the image, targets dictionary and image id
        return image, targets, image_id

    # __len__ is used with data loader for feeding data into model
    def __len__(self):
        return self.image_id.shape[0]


if __name__ == '__main__':

    # Can comment out this line if you want warnings, they became annoying
    # Only warning was from pytorch about meshgrid and how it will change in future updates
    warnings.filterwarnings("ignore")

    train_dataframe = pd.read_csv("dataset/Train.csv")

    save_figure_for_class_distribution = False

    # Will use seaborn and matplotlib to construct a class distribution figure and save it
    if save_figure_for_class_distribution:
        sns.countplot(x='class', data=train_dataframe)
        plt.savefig("Class_distribution_in_data.png")

    # Map classes to numerical values instead of string literals for pytorch to use
    class2index = {
        "fruit_healthy": 1,
        "fruit_brownspot": 2,
        "fruit_woodiness": 3
    }

    idx2class = {v: k for k, v in class2index.items()}

    train_dataframe['class'].replace(class2index, inplace=True)

    image_ids_for_dataframe = train_dataframe['Image_ID'].unique()

    # Split data set into training and generalization sets
    train_ids = image_ids_for_dataframe[:2700]
    valid_ids = image_ids_for_dataframe[2700:]

    train_dataframeSplit = train_dataframe[train_dataframe['Image_ID'].isin(train_ids)]

    valid_dataframe = train_dataframe[train_dataframe['Image_ID'].isin(valid_ids)]

    # Creates two data sets that will look in the directories passed to it for images
    train_dataset = passion_fruit_dataset(train_dataframeSplit, "dataset/Train_Images/")

    validation_dataset = passion_fruit_dataset(valid_dataframe, "dataset/Train_Images/")

    # custom function used for batches when using data loaders
    def collate_fn(batch):
        return tuple(zip(*batch))


    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=0,
                                   collate_fn=collate_fn
                                   )

    validation_data_loader = DataLoader(dataset=validation_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=collate_fn
                                        )
    # Set to True for training mode and False for inference or testing mode
    training_mode = False

    # Number of runs, number of models to make, each model wiil be saved with the number_of_runs as an id
    number_of_runs = 1

    number_of_epochs = 20

    # Same as number_of_runs but for inference
    model_number_for_validation = 1

    # Set device for pytorch to use GPU or CPU, which ever is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if training_mode:

        while number_of_runs < -1:
            try:
                print("Attempting to load model number {} for more training".format(number_of_runs))
                model = torch.load("model/fasterrcnn_mobilenet_v3_large_320_fpn_custom_{}.pth".format(number_of_runs))
                print("Loaded model {}".format(number_of_runs))
            except FileNotFoundError:
                print("Saved model not found, creating new model")

                model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
                num_classes = 4
                input_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
                print("Done creating new model")

            print("Initializing optimizer for training")

            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005)

            print("Done initialization for optimizer for training")

            model.to(device)
            model.train()

            print("Beginning training")

            for epoch in range(number_of_epochs):

                iteration_number = 1

                loss_classifier_total = 0
                loss_box_reg_total = 0
                loss_objectness_total = 0
                loss_rpn_box_reg_total = 0

                for images, targets, image_id in train_data_loader:
                    images = list(image.to(device) for image in images)

                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)

                    loss_classifier_total += loss_dict["loss_classifier"]
                    loss_box_reg_total += loss_dict["loss_box_reg"]
                    loss_objectness_total += loss_dict["loss_objectness"]
                    loss_rpn_box_reg_total += loss_dict["loss_rpn_box_reg"]

                    losses = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    iteration_number += 1

                stringOutput = f"Epoch: {epoch}\n" f"loss_classifier: {(loss_classifier_total / iteration_number)}\n" f"loss_box_reg: {(loss_box_reg_total / iteration_number)}\n" f"loss_objectness: {(loss_objectness_total / iteration_number)}\n" f"loss_rpn_box_reg: {(loss_rpn_box_reg_total / iteration_number)}\n"

                print()
                print(f"Epoch: {epoch}")
                print(f"loss_classifier: {(loss_classifier_total / iteration_number)}")
                print(f"loss_box_reg: {(loss_box_reg_total / iteration_number)}")
                print(f"loss_objectness: {(loss_objectness_total / iteration_number)}")
                print(f"loss_rpn_box_reg: {(loss_rpn_box_reg_total / iteration_number)}")

                # Saves outputs into a .txt file for later use
                with open("./RCNN_train_and_val_results/output_training_{}.txt".format(number_of_runs), "a") as f:
                    f.write(stringOutput)

            # Save model with number_of_runs as an id
            torch.save(model, "model/fasterrcnn_mobilenet_v3_large_320_fpn_custom_{}.pth".format(number_of_runs))

            number_of_runs += 1
    else:

        while model_number_for_validation < -1:

            try:
                print("Attempting to load model number {} for validation".format(model_number_for_validation))
                model = torch.load("model/fasterrcnn_mobilenet_v3_large_320_fpn_custom_{}.pth".
                                   format(model_number_for_validation))
                print("Loaded model {}".format(model_number_for_validation))
            except FileNotFoundError:
                print("File not found, exiting problem since model was not initialized")
                exit(0)

            print("Beginning validation")
            with torch.no_grad():
                model.eval()

                total = 0
                sum_loss = 0
                correct = 0

                for images, target, image_id in validation_data_loader:
                    targeted_total = target[0]['labels'].shape[0]
                    images = list(img.to(device) for img in images)
                    target = [{k: v.to(device) for k, v in t.items()} for t in target]

                    outputs = model(images)

                    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

                    predicted_boxes = copy.deepcopy(outputs[0]['boxes'].cpu().numpy().astype(np.int32))
                    target_boxes = copy.deepcopy(target[0]['boxes'].cpu().numpy().astype(np.int32))

                    predicted_labels = copy.deepcopy(outputs[0]['labels'].cpu().numpy().astype(np.int32))
                    target_labels = copy.deepcopy(target[0]['labels'].cpu().numpy().astype(np.int32))

                    predicted_scores = copy.deepcopy(outputs[0]['scores'].cpu().numpy().astype(np.float32))

                    # Had problems where there were either too few or too many predicted bounding boxes by model
                    # This if and elif will adjust the boxes, labels, and scores so that metrics can be extracted
                    # from outputs
                    if len(predicted_boxes) < len(target_boxes):
                        target_length = len(target[0]['boxes'].cpu().numpy().astype(np.int32))
                        while len(predicted_boxes) != len(target_boxes):
                            temp = np.array([[0, 0, 0, 0]])
                            predicted_boxes = np.append(predicted_boxes, temp, axis=0)
                            predicted_labels = np.append(predicted_labels, 0)
                            predicted_scores = np.append(predicted_scores, 0)

                    elif len(predicted_boxes) > len(target_boxes):
                        target_length = len(target[0]['boxes'].cpu().numpy().astype(np.int32))

                        predicted_boxes = predicted_boxes[:target_length]
                        predicted_labels = predicted_labels[:target_length]
                        predicted_scores = predicted_scores[:target_length]

                    predicted_boxes = torch.from_numpy(predicted_boxes)
                    target_boxes = torch.from_numpy(target_boxes)

                    predicted_labels = torch.from_numpy(predicted_labels)
                    target_labels = torch.from_numpy(target_labels)

                    # Gather the loss from predicted bounding boxes and gather number of correct predictions
                    loss_bb = Functional.l1_loss(predicted_boxes, target_boxes, reduction="none").sum(1)
                    loss_bb = loss_bb.sum()
                    loss = loss_bb / 1000
                    correct += predicted_labels.eq(target_labels).sum().item()
                    sum_loss += loss.item()
                    total += targeted_total

                    sample_clone = np.array(torchvision.transforms.ToPILImage()(images[0]))

                    predicted_boxes = outputs[0]['boxes'].cpu().numpy().astype(np.int32)

                    predicted_labels = outputs[0]['labels'].cpu().numpy().astype(np.int32)

                    predicted_scores = outputs[0]['scores'].cpu().numpy().astype(np.float32)

                    # Below constructs images, these will be the images used in inference mode.
                    # The images contain bounding boxes which is either in green, blue or red.
                    # Green means a healthy passion fruit.
                    # Blue means a passion fruit which has brownspots
                    # Red means a passion fruit which is woody
                    # The associated confidence levels for each bounding box is also layered on the
                    # top right of each image
                    classIndex = 0
                    font_size = 0.5
                    font_thickness = 1
                    x = 440
                    y = 20
                    for box in predicted_boxes:
                        if predicted_labels[classIndex] == 1:
                            cv2.rectangle(sample_clone,
                                          (box[0], box[1]),
                                          (box[2], box[3]),
                                          (0, 255, 0), thickness=2)
                            font_color = (0, 255, 0)
                            img_text = cv2.putText(sample_clone, str(predicted_scores[classIndex].round(4)),
                                                   (x, y + (classIndex * 20)),
                                                   cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color,
                                                   font_thickness, cv2.LINE_AA)
                        elif predicted_labels[classIndex] == 2:
                            cv2.rectangle(sample_clone,
                                          (box[0], box[1]),
                                          (box[2], box[3]),
                                          (0, 0, 255), thickness=2)
                            font_color = (0, 0, 255)
                            img_text = cv2.putText(sample_clone, str(predicted_scores[classIndex].round(4)),
                                                   (x, y + (classIndex * 20)),
                                                   cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color,
                                                   font_thickness, cv2.LINE_AA)
                        elif predicted_labels[classIndex] == 3:
                            cv2.rectangle(sample_clone,
                                          (box[0], box[1]),
                                          (box[2], box[3]),
                                          (255, 0, 0), thickness=2)
                            font_color = (255, 0, 0)
                            img_text = cv2.putText(sample_clone, str(predicted_scores[classIndex].round(4)),
                                                   (x, y + (classIndex * 20)),
                                                   cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color,
                                                   font_thickness, cv2.LINE_AA)
                        else:
                            cv2.rectangle(sample_clone,
                                          (box[0], box[1]),
                                          (box[2], box[3]),
                                          (255, 255, 255), thickness=2)
                            font_color = (255, 255, 255)
                            img_text = cv2.putText(sample_clone, str(predicted_scores[classIndex].round(4)),
                                                   (x, y + (classIndex * 20)),
                                                   cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color,
                                                   font_thickness, cv2.LINE_AA)
                        classIndex += 1

                    image_to_save = Image.fromarray(sample_clone)

                    # Will save images to a directory corresponding to a folder in the result directory with the
                    # same model number
                    # Not a very elegant solution but it works and keeps each 301 images for each model nicely
                    # organized
                    image_to_save.save("result/{}/{}_result_{}.jpg".format(model_number_for_validation, model_number_for_validation, image_id[0]))

                stringOutput = "Model number: " + str(model_number_for_validation) + "\n" + "Sum-loss / total: " + str(sum_loss / total) + "\n" + "Correct / total: " + str(correct / total) + "\n"

                # Saves outputs in .txt file for later use
                # Will append to file, so if a file already exists from previous experimentation then delete that one first
                with open("./RCNN_train_and_val_results/output_validation_{}.txt".format(model_number_for_validation), "a") as f:
                    f.write(stringOutput)

                print("Sum-loss: ", sum_loss)
                print("Correct: ", correct)
                print("Total: ", total)
                print()
                print("Sum-loss / total and correct / total")
                print(sum_loss / total, correct / total)

                model_number_for_validation += 1
