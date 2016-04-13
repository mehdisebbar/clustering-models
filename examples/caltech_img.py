from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

from K_estim_pi_pen_EM import GraphLassoMix

orb = cv2.ORB(nfeatures=100)
images = [f for f in listdir('./imgs/') if isfile(join('./imgs/', f))]
descriptor_dataset = []
for img_path in images:
    img = cv2.imread("./imgs/" + img_path)
    resized = cv2.resize(img, (500,500), interpolation = cv2.INTER_AREA)
    kp = orb.detect(resized,None)
    _, des = orb.compute(resized, kp)
    descriptor_dataset.append(np.sort(np.linalg.norm(des,axis=1)))
desc_data = np.array(descriptor_dataset)


min_desc = 1000
for d in desc_data:
    if len(d)<min_desc:
        min_desc = len(d)
a = []
for d in desc_data:
    a.append(d[:min_desc])
desc_data = np.array(a)

cl = GraphLassoMix(lambda_param=0.005, n_iter=15, max_clusters=15)
cl.fit(desc_data)
