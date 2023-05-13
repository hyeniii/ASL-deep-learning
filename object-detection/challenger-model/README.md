# ASL Object Detection Challenger Model

## Directory Structure in Deepdish

- Root

![](./images/challenger-model.png)

- Annotations

![](./images/annotations.png)

- Data

![](./images/data.png)

- Roboflow

![](./images/roboflow.png)

- Raw
    - The processed directory has the same structure as below but contains images that are trimmed based on the bounding box coordinates before training the model. 

![](./images/raw.png)

- Train 
    - The validation directory has the same structure as train
    - The test directory **doesn't** contain subdirectories like train and validation and instead only contains the images.

![](./images/train.png)
