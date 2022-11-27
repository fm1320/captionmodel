from pathlib import Path
 
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from sentence_transformers import SentenceTransformer
from PIL import Image
 
def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
 
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
 
    if len(a.shape) == 1:
       a = a.unsqueeze(0)
 
    if len(b.shape) == 1:
       b = b.unsqueeze(0)
 
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
 
class Task_Model(nn.Module):
    def __init__(self, max_length:int=16, num_beams:int=4):
        super().__init__()
        self.sententece_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cap_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.cap_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.cap_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
 
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
 
    def forward(self, img_input, text_input):
        pixel_values = self.cap_feature_extractor(images=img_input, return_tensors="pt").pixel_values
        output_ids = self.cap_model.generate(pixel_values, **self.gen_kwargs)
 
        captions = self.cap_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        captions = [caption.strip() for caption in captions]
        
        embeddings1 = self.sententece_encoder.encode(captions, convert_to_tensor=True)
        embeddings2 = self.sententece_encoder.encode(text_input, convert_to_tensor=True)
        similarity_scores = cos_sim(embeddings1, embeddings2)
        np_array = similarity_scores.cpu().numpy() # the tensor output is not serializable
        return (dict(enumerate(np_array.flatten(), 1)))
 
def process_image(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")
 
    images.append(i_image)
  return images
 
model_obj = Task_Model()

__version__ = "0.1.0" 
BASE_DIR = Path(__file__).resolve(strict=True).parent 
PATH =f"{BASE_DIR}/model_{__version__}.pt"

torch.save(model_obj.state_dict(), PATH) 
model_obj.load_state_dict(torch.load(PATH))
 
images1= process_image([f"{BASE_DIR}/img_1.jpg"]) 
 
sentences2 = ['Car driving outside',
              'A girl playing tennis',
              'The dog running ']          # This are just example sentences that are compared to the image captioon
              
result=model_obj.forward(images1,sentences2)

# Prints the cosine similarity 
#print(result) 

 

 

