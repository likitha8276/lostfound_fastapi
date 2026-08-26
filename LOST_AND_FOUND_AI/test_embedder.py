import torch
from transformers import AutoTokenizer,AutoModel,CLIPProcessor,CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
import io

print("Loading ML Models(MPNEY & CLIP)...")

mpnet_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
mpnet_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")


clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

print("Models loaded successfully")

sentence1 = "black leather wallet with cards"
sentence2 = "found dark brown wallet"

tokens = mpnet_tokenizer([sentence1,sentence2],padding=True,truncation=True,return_tensors="pt")
print(tokens)


