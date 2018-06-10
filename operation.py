
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch
import cv2
from model import EncoderCNN, DecoderRNN
import argparse

def parse_parameters():
    parser = argparse.ArgumentParser(description='Image_caption')

    parser.add_argument("--input", dest = 'img', help =
                        "Image / Directory containing images", type = str)
    parser.add_argument("--show", dest = 'show', help =
                        "if you will show the input image,set 1", type = int)

    return parser.parse_args()

def clean_sentence(output):
    sentence = ""
    for word_token in output:
        if word_token == 0 or word_token == 2:
            continue
        if word_token == 1 :
            return sentence
        sentence += (data_loader.dataset.vocab.idx2word[word_token] + " ") 
    return sentence

def get_prediction():
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)    
    sentence = clean_sentence(output)
    print(sentence)

def checkTheInput(img):
    plt.imshow(img)
    plt.title('test image')
    plt.show()


def get_model(device,vocab_size):
    # model weights file
    encoder_file = "models/encoder-3.pkl" 
    decoder_file = "models/decoder-3.pkl"

    embed_size = 512
    hidden_size = 512

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Load the trained weights.
    #print(torch.load(encoder_file))
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))

    # Move models to GPU if CUDA is available.
    encoder.to(device)
    decoder.to(device)

    return encoder,decoder

def caption(img,device,vocab_size):
    # Move image Pytorch Tensor to GPU if CUDA is available.
    image = img.to(device)
    # model
    encoder,decoder = get_model(device,vocab_size)
    # img_embed
    features = encoder(image).unsqueeze(1)
    # Pass the embedded image features through the model to get a predicted caption.
    output = decoder.sample(features)
    # sentence
    plt.imshow(np.squeeze(image))
    plt.title('Sample Image')
    plt.show()
    sentence = clean_sentence(output)
    print(sentence)

# Main
parameters = parse_parameters()

image = parameters.img 

isShow = parameters.show

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# transform_test = transforms.Compose([ 
# transforms.Resize((224,224)), # smaller edge of image resized to 224
# transforms.RandomCrop(224), 
# transforms.ToTensor(),                           # convert the PIL Image to a tensor
# transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
#                          (0.229, 0.224, 0.225))])

# data_loader = get_loader(transform=transform_test,    
#                          mode='test')

vocab_size = 9986#len(data_loader.dataset.vocab)

if image is None:
    orig_image, image = next(iter(data_loader))
else:
    image = (cv2.imread(image)[:,:,::-1]).astype(np.float)
    # resize
    image = cv2.resize(image,(224,224))
    # Normalize
    image /= 255.0
    image -= (0.485, 0.456, 0.406)
    image /= (0.229, 0.224, 0.225)
    # To tensor
    image = torch.from_numpy(image)

# show test image
if isShow:
    checkTheInput(image)

# caption
caption(image,device,vocab_size)








