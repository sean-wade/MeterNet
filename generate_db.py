import os
import torch

from MeterNet.model import FaceModelCenter
from MeterNet.features import Extractor
from database.faceSQLite import FaceSQL


DB_NAME = "Meter.db"
if os.path.exists(DB_NAME): 
    os.remove(DB_NAME)


extractor = Extractor(checkpoint='./weights/checkpoint_49.pth')
#extractor = Extractor(center_type = False, embedding_size=256, checkpoint='./weights/checkpoint_95.pth')    #Not good

faceDB = FaceSQL(DB_NAME)


imgs = os.listdir('./img')
for i,img in enumerate(imgs):
    feature = extractor.get_feature('./img/' + img)
    #print(feature)
    feature_str = Extractor.vector_to_str(feature)
    #print(len(feature_str))
    faceDB.insert(i, img.split('.')[0], feature_str)


print(faceDB.queryID(4))
#print(faceDB.queryAll())


faceDB.close()
