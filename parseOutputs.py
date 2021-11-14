# This file reads .txt files saved from training and inference to perform statistical analysis on
# Is hard coded to get statistics from 30 models which ran for 20 epochs
# Is also hard coded to follow naming conventions of files in main.py

# Change values if used other parameters

import statistics

outputNumber = 1

loss_classifier_list = []
loss_box_reg_list = []
loss_objectness_list = []
loss_rpn_box_reg_list = []

while outputNumber < 31:
    fileTraining = open("RCNN_train_and_val_results/output_training_{}.txt".format(outputNumber))

    last_metric = False

    line = fileTraining.readline()
    while line != "":
        line = fileTraining.readline()
        if line.startswith("Epoch: 19"):
            last_metric = True

        if last_metric:
            if line.startswith("loss_classifier"):
                line = line.split()
                loss_classifier_list.append(float(line[1]))
            elif line.startswith("loss_box_reg"):
                line = line.split()
                loss_box_reg_list.append(float(line[1]))
            elif line.startswith("loss_objectness"):
                line = line.split()
                loss_objectness_list.append(float(line[1]))
            elif line.startswith("loss_rpn_box_reg"):
                line = line.split()
                loss_rpn_box_reg_list.append(float(line[1]))

    last_metric = False

    outputNumber += 1

print("loss_classifier_list")
print("Min: ", min(loss_classifier_list))
print("Mean: ", statistics.mean(loss_classifier_list))
print("Standard deviation: ", statistics.stdev(loss_classifier_list))
print()
print("loss_box_reg_list")
print("Min: ", min(loss_box_reg_list))
print("Mean: ", statistics.mean(loss_box_reg_list))
print("Standard deviation: ", statistics.stdev(loss_box_reg_list))
print()
print("loss_objectness_list")
print("Min: ", min(loss_objectness_list))
print("Mean: ", statistics.mean(loss_objectness_list))
print("Standard deviation: ", statistics.stdev(loss_objectness_list))
print()
print("loss_rpn_box_reg_list")
print("Min: ", min(loss_rpn_box_reg_list))
print("Mean: ", statistics.mean(loss_rpn_box_reg_list))
print("Standard deviation: ", statistics.stdev(loss_rpn_box_reg_list))


outputNumberValidation = 1

sum_loss_average_list = []
correct_average_list = []

while outputNumberValidation < 31:
    fileValidation = open("RCNN_train_and_val_results/output_validation_{}.txt".format(outputNumberValidation))

    line = fileValidation.readline()  # Just to skip the model number in .txt file

    line = fileValidation.readline()
    line = line.split()

    sum_loss_average_list.append(float(line[3]))

    line = fileValidation.readline()
    line = line.split()

    correct_average_list.append(float(line[3]))

    outputNumberValidation += 1

print()
print("sum_loss_average_list")
print("Min: ", min(sum_loss_average_list))
print("Mean: ", statistics.mean(sum_loss_average_list))
print("Standard deviation: ", statistics.stdev(sum_loss_average_list))
print()
print("correct_average_list")
print("Max: ", max(correct_average_list))
print("Mean: ", statistics.mean(correct_average_list))
print("Standard deviation: ", statistics.stdev(correct_average_list))
