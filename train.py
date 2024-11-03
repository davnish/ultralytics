from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.pt")

# Train the model
train_results = model.train(
    data="palm_trees.yaml",  # path to dataset YAML
    epochs=1000,  # number of training epochs
    imgsz=256,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("preprocessing_data/8bit/patched_full/patched_full.27.tif")
print(results[0].show())
path = model.export(format="torchscript")  # return path to exported model