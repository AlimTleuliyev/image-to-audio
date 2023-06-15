import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_url = 'https://www.shutterstock.com/image-photo/students-listening-female-teacher-classroom-260nw-778983088.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# raw_image = Image.open('Visit-the-Zoo-Day.jpg').convert('RGB')
# conditional image captioning
# text = "a photography of"
# inputs = processor(raw_image, text, return_tensors="pt")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))
# # >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
