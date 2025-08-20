import io
import numpy
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
from src.model_architecture import FasterRCNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2
model_class = FasterRCNNModel(num_classes, device)
model = model_class.model
checkpoint = torch.load("artifacts/model/F_RCNN.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

app = FastAPI()

def predict_and_draw(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(img_tensor)

    pred = predictions[0]
    box = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    for box, score in zip(box, scores):
        if score>0.7:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    return img_rgb

@app.get("/")
def root():
    return "{message: Welcom to the Guns Object Detection}"

@app.post("/predict/")
async def predict(file:UploadFile=File(...)):
    img_data = await file.read()
    image = Image.open(io.BytesIO(img_data))
    out = predict_and_draw(image)
    out_format = io.BytesIO()
    out.save(out_format, format='PNG')
    out_format.seek(0)

    return StreamingResponse(out_format, media_type="image/png")


            


