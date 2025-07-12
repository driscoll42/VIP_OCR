import os
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from craft_text_detector import Craft


def load_image(path):
    """
    Loading image
    """
    return cv2.imread(path)


def process_image(img):
    """
    Process tilted images
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)

    (cx, cy), (w, h), ang = ret
    if w > h:
        w, h = h, w
        ang -= 90

    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return rotated


def load_models(trocr_model="microsoft/trocr-large-handwritten",
                link_threshold=0.1,
                text_threshold=0.3,
                crop_type="box",
                output_dir=None):
    """
    Loading TrOCR model which has achieved SOTA metrics on IAM handwriting dataset
    """
    processor = TrOCRProcessor.from_pretrained(trocr_model)
    model = VisionEncoderDecoderModel.from_pretrained(trocr_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    craft = Craft(output_dir=output_dir,
                  crop_type=crop_type,
                  export_extra=False,
                  link_threshold=link_threshold,
                  text_threshold=text_threshold,
                  cuda=torch.cuda.is_available())

    return processor, model, craft


def detection(img, craft):
    """
    Text detection using CRAFT text detector
    """

    prediction_result = craft.detect_text(img)
    return img, prediction_result


def recoginition(img, prediction_result, processor, model, verbose=False):
    """
    OCR using TrOCR
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = []
    for i, j in enumerate(prediction_result['boxes']):
        try:
            roi = img[int(prediction_result['boxes'][i][0][1]): int(prediction_result['boxes'][i][2][1]),
                  int(prediction_result['boxes'][i][0][0]): int(prediction_result['boxes'][i][2][0])]
            image = Image.fromarray(roi).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text.append(generated_text)
            if verbose: print('line ' + str(i) + ' has been recoginized')
        except Exception as e:
            print(e)

    return prediction_result['boxes'], ('\n').join(text)


def visualize(img, prediction_result):
    for i, j in enumerate(prediction_result['boxes']):
        y1 = int(prediction_result['boxes'][i][0][1])
        y2 = int(prediction_result['boxes'][i][2][1])

        x1 = int(prediction_result['boxes'][i][0][0])
        x2 = int(prediction_result['boxes'][i][2][0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return Image.fromarray(img)
