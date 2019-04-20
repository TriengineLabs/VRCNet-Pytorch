import torch
import torch.nn as nn
import skimage
import numpy as np

class ToTensor(nn.Module):
    def __init__(self):
        super(ToTensor, self).__init__()
    
    def forward(self, tracks):
        with torch.no_grad():
            for i, el in enumerate(tracks):
                tracks[i] = torch.Tensor(el)
        return tracks


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, tracks):
        with torch.no_grad():
            vector = tracks[0]
            # vector = torch.Tensor(vector)
            min_v = np.min(vector)
            range_v = np.max(vector) - min_v
            if range_v > 0:
                normalised = (vector - min_v) / range_v
            else:
                normalised = np.zeros(vector.shape)
            tracks.insert(0, normalised)
        return tracks


class HorizontalCrop(nn.Module):
    def __init__(self, crop_size):
        super(HorizontalCrop, self).__init__()
        self.crop_size = crop_size

    def forward(self, vector):
        processed_tracks = []
        with torch.no_grad():
            for track in vector:
                cropped_track = track[:, :self.crop_size]
                processed_tracks.append(cropped_track)
        return processed_tracks


class Resize(nn.Module):
    def __init__(self, width, height):
        super(Resize, self).__init__()
        self.width = width
        self.height = height

    def forward(self, vector):
        processed_tracks = []
        with torch.no_grad():
            for track in vector:
                temp = skimage.transform.resize(track, (self.width, self.height))
                processed_tracks.append(torch.Tensor(temp))
        return processed_tracks

