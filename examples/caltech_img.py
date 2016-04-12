import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy import ndimage

orb = cv2.ORB(nfeatures=10)
images = [f for f in listdir('/home/mehdi/projects/phd/examples/imgs/') if isfile(join('/home/mehdi/projects/phd/examples/imgs/', f))]
descriptor_dataset = []
for img_path in images:
    img = cv2.imread("/home/mehdi/projects/phd/examples/imgs/"+img_path)
    resized = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
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

from K_estim_pi_pen_EM import GraphLassoMix

cl = GraphLassoMix(lambda_param=0.01, n_iter=15, max_clusters=6)
cl.fit(desc_data)
