import torch
from transformers import logging
from torch.utils.data import Dataset
import numpy as np
class BaseDataset(Dataset):
    def __init__(self, data, labels, clean_data=None):
        self.data = data
        self.labels = labels
        self.clean_data = clean_data
        logging.set_verbosity_error()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = {
            'text_bert': torch.Tensor(self.data['text'][idx]),
            'audio': torch.Tensor(self.data['audio'][idx]),
            'vision': torch.Tensor(self.data['vision'][idx]),
            'text_len': torch.LongTensor([self.data['text_len'][idx]]),
            'audio_len': torch.LongTensor([self.data['audio_len'][idx]]),
            'vision_len': torch.LongTensor([self.data['vision_len'][idx]]),
            'label': torch.Tensor(self.labels[idx].reshape(-1))
        }

        if self.clean_data != None:
            if 'id' in self.data:
                index = np.squeeze(np.where(self.clean_data['id'] == self.data['id'][idx]))
                sample_clean = {
                    'clean_text_bert': torch.Tensor(self.clean_data['text'][index]),
                    'clean_audio': torch.Tensor(self.clean_data['audio'][index]),
                    'clean_vision': torch.Tensor(self.clean_data['vision'][index]),
                    'clean_label': torch.Tensor(self.clean_data['label'][index].reshape(-1))
                }
            sample.update(sample_clean)
        
        return sample
    
class NoiseCleanDataset(Dataset):
    def __init__(self, clean_data, noise_data):
        self.clean_data = clean_data
        self.noise_data = noise_data
        self.labels = clean_data['label']
        logging.set_verbosity_error()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        noise_clean_data = {
            'clean_text_bert': torch.Tensor(self.clean_data['text'][idx]),
            'clean_audio': torch.Tensor(self.clean_data['audio'][idx]),
            'clean_vision': torch.Tensor(self.clean_data['vision'][idx]),
            'label': torch.Tensor(self.labels[idx].reshape(-1)),
            'noise_text_bert': torch.Tensor(self.noise_data['text'][idx]),
            'noise_audio': torch.Tensor(self.noise_data['audio'][idx]),
            'noise_vision': torch.Tensor(self.noise_data['vision'][idx]),
            'text_len': torch.LongTensor([np.array(self.clean_data['text_len'])[idx]]),
            'audio_len': torch.LongTensor([np.array(self.clean_data['audio_len'])[idx]]),
            'vision_len': torch.LongTensor([np.array(self.clean_data['vision_len'])[idx]]),
        }
        return noise_clean_data
    

    