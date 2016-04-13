import itertools
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join

import cv2
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from K_estim_pi_pen_EM import GraphLassoMix
from tools.gm_tools import best_cont_matrix


def eval_caltech_img(nfeatures, lambda_param, max_clusters):
    orb = cv2.ORB(nfeatures=nfeatures)
    images = [f for f in listdir('./imgs/') if isfile(join('./imgs/', f))]
    descriptor_dataset = []
    for img_path in images:
        img = cv2.imread("./imgs/" + img_path)
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
    sc = StandardScaler()
    cl = GraphLassoMix(lambda_param=lambda_param, n_iter=15, max_clusters=max_clusters)
    pi, y, means, covars = cl.fit(sc.fit_transform(desc_data))
    return {"labels": y, "params": (nfeatures, lambda_param, max_clusters)}


def multiple_arg_parser_caltech(t):
    nfeatures, lambda_param, max_clusters = t[0], t[1], t[2]
    return eval_caltech_img(nfeatures, lambda_param, max_clusters)

if __name__ == '__main__':
    param_list = [[50, 100, 200], [0.01, 0.1, 1, 10], [5, 10]]  # nfeatures, lambda_param, max_clusters
    params_comb = list(itertools.product(*param_list))
    p = Pool(joblib.cpu_count())
    res = p.map(multiple_arg_parser_caltech, params_comb)
    images = [f for f in listdir('./imgs/') if isfile(join('./imgs/', f))]
    y_real = [int(img.split("_")[0]) for img in images]
    for r in res:
        print r["params"]
        print best_cont_matrix(r["labels"], y_real)
