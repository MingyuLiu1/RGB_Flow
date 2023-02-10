import os
import numpy as np
import pickle
import time

gtPath = 'kitti_rm_ground' # ground truth
predictionBasePath = 'saved_flow_eval' # our prediction scene flow from voxel_encoder.py

def calculate_endPointError(gt, pd):
    l2_norm = np.linalg.norm(gt - pd, axis=-1)
    EPE3D = l2_norm.mean()
    return EPE3D


def filename_convention(file):
    file = file.__str__()
    while len(file) < 6:
        file = "0" + file
    return file


def create_custom_dir(file):
    if not os.path.exists(file):
        os.makedirs(file)
        print("The new " + file + " is created!")


def findIndexInGT(first_sample_point_coors, gtFirstScene):
    return np.where(gtFirstScene == first_sample_point_coors)


def find_closest(arr, val):
    l2_norm = np.linalg.norm(arr - val, axis=-1)
    idx = l2_norm.argmin()
    return idx

totalSize = 300
pcSize = 2048
fend = open(("eval_results"+".txt"), 'w')
for k in range(1,2):
    predictionPath = predictionBasePath + "_" + str(k)
    fend.write(str(k))
    fend.write("\n")
    epochResult = 0
    epochTotal = 0

    for n in range(1,2):
        iterResult = 0
        iterTotal = 0
        #start = time.time()
        predictionIterPath = predictionPath + "_" + str(n)
        for i in range(0, totalSize, 2):

            f1 = os.path.join(predictionIterPath, (filename_convention(i) + ".npy"))
            predictionContent = pickle.load(open(f1, 'rb'))
            first_sample_point_coors = predictionContent["first_sample_point_coors"][0].numpy().transpose()
            predicted_flow = predictionContent["predicted_flow"][0].numpy().transpose()

            f2 = os.path.join(gtPath, (filename_convention(int(i/2)) + ".npz"))
            groundtruthContent = np.load(f2)
            gtFirstScene = groundtruthContent['pos1']
            gtSecondScene = groundtruthContent['pos2']
            gtSFlow = groundtruthContent['gt']
            sceneResult = 0
            for j in range(pcSize):
                # find index of the j'th sample point in ground truth file
                index = find_closest(gtFirstScene,first_sample_point_coors[j])
                epe = calculate_endPointError(gtSFlow[index],predicted_flow[j])
                sceneResult += epe
            iterTotal += sceneResult

        iterResult = iterTotal / (pcSize*150)
        epochTotal += iterResult

        #end = time.time()
        #timeDiff = end-start
        #print(timeDiff)
        print("Epoch: {} Iter: {} Results: {}".format(k,n,iterResult))
        fend.write("Epoch: {} Iter: {} Results: {}".format(k,n,iterResult))
        fend.write("\n")
    epochResult = epochTotal / 10
    print(epochResult)
    fend.write("Epoch {} : Average End Point Error: {}".format(k,epochResult))
    fend.write("\n")

fend.close()
