import torch
from PIL import Image
import open_clip
import torch.nn.functional as F


class ClipExtract(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _, preprocess = open_clip.create_model_and_transforms('convnext_base', pretrained='laion400m_s13b_b51k')

        #tokenizer = open_clip.get_tokenizer('convnext_base')
        #self.model, _, preprocess = open_clip.create_model_and_transforms('RN50x4', pretrained='openai')
        #self.model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        
        
    def forward(self, x, y):
        # 224x224 -> 768 embeddings
        x = F.interpolate(x, (244, 244), mode="bilinear")  
        x = self.model.encode_image(x)

        y = F.interpolate(y, (244, 244), mode="bilinear")  
        y = self.model.encode_image(y)

        return (1 / (F.cosine_similarity(x, y).mean()-1))