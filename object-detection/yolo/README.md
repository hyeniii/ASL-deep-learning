# Training YOLO on ASL-Data
Step-by-step guide on how to train custom model using Pytorch-based YOLOv5 from Ultralytics.

## Step 0: Download Dataset

https://public.roboflow.com/object-detection/american-sign-language-letters

download dataset for yolov5

![yolo-data-directory]('./imgs/yolo-data-dir.png')

## Step 1: Download yolo repository from github
```bash
git clone https://github.com/ultralytics/yolov3.git
```

## Step 2: Set up virtual enivronment (Optional)
```bash
# create virtual environment
python -m venv .venv

# activate virtual environment
source .venv/bin/activate

# donwload required packages
pip install -r requirements.txt
```

## Step 3: Create YAML file for the custom dataset
```yaml
train: <PATH TO TRAIN>
val: <PATH TO VALIDATION>

nc: <NUMBER OF CLASSES>
names: <LIST OF CLASS NAMES>
```

## Step 4: Train Yolo on ASL-Data using Yolov5m weights
- set image size to 640
- set epoch to 100
- activate Comet to log training process and metrics

```bash
export COMET_API_KEY= <YOUR-COMET-API-KEY>
python train.py --img 640 --epochs 100 --data asl-data/data.yaml --weights yolov5m.pt
```

## Step 5: Validate trained model
```bash
python val.py --img 640 --data asl-data/data.yaml --weights runs/train/exp6/weights/best.pt
```

## Step 6: Run real-time predictions using web-camera
```bash
python detect.py --weights runs/train/exp6/weights/best.pt --source 0  
```