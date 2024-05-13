from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection, AutoModelForObjectDetection
from paddleocr import PaddleOCR
import numpy as np
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.on_event("startup")
async def startup_event():
    app.state.image_processor = AutoImageProcessor.from_pretrained("bilguun/table-transformer-structure-recognition")
    app.state.model = AutoModelForObjectDetection.from_pretrained("bilguun/table-transformer-structure-recognition")
    app.state.model.cuda().eval()  
    app.state.ocr = PaddleOCR(lang='en', debug=False, show_log=False, use_angle_cls=False)


@app.on_event("shutdown")
async def shutdown_event():
    del app.state.image_processor
    del app.state.model
    del app.state.ocr


def convert_to_bbox(vertices):
    xs, ys = zip(*vertices)
    return [min(xs), min(ys), max(xs), max(ys)]


def detect_merged(image_input: np.ndarray, threshold_conf: float = 0.5) -> str:
    try:
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 4:
                image_array = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 1:
                image_array = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 2:
                image_array = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError("Unsupported NumPy array format")
        elif isinstance(image_input, Image.Image):
            image_array = np.array(image_input)
        else:
            raise ValueError("Input must be a NumPy array or PIL Image object")

        image_pil = Image.fromarray(image_array)

        inputs = app.state.image_processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}  

        outputs = app.state.model(**inputs)
        target_sizes = torch.tensor([image_pil.size[::-1]]).cuda() 
        results = app.state.image_processor.post_process_object_detection(outputs, threshold=threshold_conf, target_sizes=target_sizes)[0]

        bbox_struct_list = [[round(i, 2) for i in box.tolist()] for box in results["boxes"]]

        text_detections = app.state.ocr.ocr(image_array, cls=False)
        if not text_detections:
            return "No text detected."

        text_detections = [item for sublist in text_detections for item in sublist]  
        text_boxes = [convert_to_bbox(detection[0]) for detection in text_detections if detection]

        response = "unmerged table"
        for table in bbox_struct_list:
            for text in text_boxes:
                if ((table[2] <= text[2] and table[2] >= text[0]) or (table[0] <= text[2] and table[0] >= text[0])) and (table[3] >= text[3] and table[1] <= text[1]):
                    response = "merged table"
                    break
            if response == "merged table":
                break

        return response

    except Exception as e:
        return f"Processing error: {str(e)}"


@app.post("/upload-array/")
async def upload_array(data: dict):
    try:
        array_data = data["array"]
        
        np_array = np.array(array_data, dtype=np.uint8)

        image_pil = Image.fromarray(np_array, 'RGB')  

        response = detect_merged(image_pil)
        return JSONResponse(content={"response": response})

    except KeyError:
        raise HTTPException(status_code=400, detail="Array data not provided.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the array: {str(e)}")