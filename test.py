from ultralytics import YOLO
import glob
import os

# Load a model
model = YOLO("runs/detect/train3/weights/best.pt")

# Set the model to validation mode
metrics = model.val()

for i in glob.glob("preprocessing_data/8bit/patched_full/*.tif"):
    file_name = os.path.splitext(os.path.split(i)[1])[0]
    results = model(i)
    results[0].show()
    print(results[0].save_txt(f'results/{file_name}.txt'))
    
