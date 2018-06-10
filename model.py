import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # not training
        # transfer learning
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # print("resnet.fc.in_features:",resnet.fc.in_features)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        # batch_size * neural_unit_count
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    # input : seq_len, batch, input_size
    # output: seq_len, batch, hidden_size * num_directions cbnm
    # hidden,C: num_layers * num_directions, batch, hidden_size,num_layers * num_directions, batch, hidden_size
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN,self).__init__()
        
        # basic dims
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # embed
        # output : batch * seq_len * embed_size
        self.embed = nn.Embedding(vocab_size,embed_size)
        # the first LSTM output:<start>
        # input is : (batch_size,len_seq,256)  output: 10,len_seq,512
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        # linear to embed_size
        self.linear = nn.Linear(hidden_size,vocab_size)
        
    # test: default is zeros    
    def _init_hidden_status(self,num_layers,features_size,hidden_size):
        return torch.zeros(num_layers,features_size,hidden_size)
    
    def forward(self, features, captions):
        # features --> Image Vetor   batch_size * 256
        # captions --> batch_size * len_sentence
        # output : batch_size * 1 * 512
        features = features.unsqueeze(1)
        # captions --> Image captions  batch * (len_seq)
        # tranform to embeding_vector
        # output : batch_size * seq_len * 256
        # The decoder will learn when to generate the <end> word.
        # don`t contain the <end>
        # To ensure the model will output the full sentence.
        # output: batch * (len_seq - 1) * embed_size
        embeding_vector = self.embed(captions[:,:-1])
        # output: batch_size * seq_len * embed_size
        embeding_vector = torch.cat((features,embeding_vector),1)
        # print("cat after embeding_vector:",embeding_vector.shape)
        # output: out--> batch_size * seq_len * 512
        out,(h,c) = self.lstm(embeding_vector) 
        out = self.linear(out)
        # batch_size * vocab_size
        return out
        
    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # the first word
        # output: 1 * 1 * 512
        output_tokens = []
        for i in range(max_len):
            # features : 1 * 1 *  512,output: 1* 1 * 512
            output,hidden = self.lstm(inputs,hidden)
            # linear, output: 1 * 1 * 9986
            output = self.linear(output.squeeze(1))
            # max_value  
            # print(torch.topk(output,10))
            max_index = output.max(1)[1]#torch.argmax(output,dim=2).item() 
            # add the predict
            output_tokens.append(max_index.item())
            # input: 1 * 1 output: 1 * 1 * 512
            inputs = self.embed(max_index).unsqueeze(1)
            
        return output_tokens
    
    # attension
    def attension(self,encoder_features,captions):
        pass
            
                