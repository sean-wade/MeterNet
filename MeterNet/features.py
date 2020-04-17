import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd import Function

from .model import FaceModelCenter, FaceModel
from .distance import PairwiseDistance


class Extractor(object):
    def __init__(self, center_type = True, embedding_size=512, num_classes=22, checkpoint='../weights/checkpoint_49.pth'):
        if center_type:        
            self.model = FaceModelCenter(embedding_size = embedding_size,
                                         num_classes    = num_classes,
                                         checkpoint     = None)
        else:
            self.model = FaceModel(embedding_size = embedding_size,
                                         num_classes    = num_classes)

        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.l2_dist = PairwiseDistance(2)

        self.transform = transforms.Compose([transforms.Resize((96, 96)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],
                                                                  std=[0.5, 0.5, 0.5])])


    @staticmethod
    def vector_to_str(vector):
        vector = list(vector)
        new_vector = [str(x) for x in vector]
        return ','.join(new_vector)


    @staticmethod
    def str_to_vector(face_str):
        str_list = face_str.split(',')
        return np.array([float(x) for x in str_list])


    def get_feature_torch(self, path_a):
        img = Image.open(path_a)
        a_v = Variable(torch.unsqueeze(self.transform(img), dim=0).float(), requires_grad=False)
        
        with torch.no_grad():
            feature = self.model(a_v)
        return feature

    
    def get_feature(self, path_a):
        return np.array(self.get_feature_torch(path_a)[0])



    def calc_distance(self, path_a, path_b):
        s1, s2 = self.get_feature_torch(path_a), self.get_feature_torch(path_b)
        return float(self.l2_dist.forward(s1,s2).data[0])


'''

from annoy import AnnoyIndex

class Annoy(object):
    def __init__(self):
        self.annoy = AnnoyIndex(512)
        self.annoy_index_file = 'face_id'
        self.update_index()

    def update_index(self):
        self.annoy.unload()
        users = User.query.all()
        for user in users:
            self.annoy.add_item(user.id, self.str_to_vector(user.faceId))
        self.annoy.build(100)
        self.annoy.save(self.annoy_index_file)
        self.annoy.load(self.annoy_index_file)

    @staticmethod
    def vector_to_str(vector):
        new_vector = [str(x) for x in vector]
        return ','.join(new_vector)

    @staticmethod
    def str_to_vector(face_str):
        str_list = face_str.split(',')
        return [float(x) for x in str_list]

    def query_face_id(self, face_id):
        return self.annoy.get_nns_by_vector(face_id, 5, include_distances=True)'''
