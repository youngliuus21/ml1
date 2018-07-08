%matplotlib inline
import cv2
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

def video_to_frames(filename, output_dir, prefix=''):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    vidcap = cv2.VideoCapture(filename)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(output_dir, prefix + '_%d.png') % count, image)
            count += 1
        else:
            break
    vidcap.release()
    
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def RotateFiles(rootdir, pattern, angle, outputdir, appendix):
    p = Path(rootdir)
    files = list(p.glob(pattern))
    for f in files:
        fname = f.name
        img = cv2.imread(os.path.join(rootdir, fname))
        res = rotateImage(img, angle)
        res_name = os.path.splitext(fname)[0] + appendix + '.png'
        cv2.imwrite(os.path.join(outputdir, res_name), res)
        # print('write img:'+res_name)
            
def ToGrayFiles(rootdir, pattern, outputdir, appendix):
    p = Path(rootdir)
    files = list(p.glob(pattern))
    for f in files:
        fname = f.name
        img = cv2.imread(os.path.join(rootdir, fname))
        res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res_name = os.path.splitext(fname)[0] + appendix + '.png'
        cv2.imwrite(os.path.join(outputdir, res_name), res)

def CropFiles(rootdir, pattern):
    p = Path(rootdir)
    files = list(p.glob(pattern))
    for f in files:
        fname = f.name
        img = cv2.imread(os.path.join(rootdir, fname))
        res = img[50:300, 100:500,:]
        cv2.imwrite(os.path.join(rootdir, fname), res)
        
def ShrinkFiles(rootdir, pattern, outputdir, appendix):
    p = Path(rootdir)
    files = list(p.glob(pattern))
    for f in files:
        fname = f.name
        img = cv2.imread(os.path.join(rootdir, fname))
        res = cv2.resize(img, (200, 125), interpolation = cv2.INTER_AREA)
        res_name = os.path.splitext(fname)[0] + appendix + '.png'
        cv2.imwrite(os.path.join(outputdir, res_name), res)
        
def ProcessImage(img):
    res = img[50:300, 100:500,:]
    res = cv2.resize(res, (200, 125), interpolation = cv2.INTER_AREA)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    return res

def GenerateData(rootdir, outputdir, pattern='*'):
    p = Path(rootdir)
    files = list(p.glob(pattern))
    
    for f in files:
        fname = f.name
        org = cv2.imread(os.path.join(rootdir, fname))
        org_s = ProcessImage(org)
        img_l = ProcessImage(rotateImage(org, 90))
        img_r = ProcessImage(rotateImage(org, -90))
        cv2.imwrite(os.path.join(outputdir, os.path.splitext(fname)[0] + '.png'), org_s)
        cv2.imwrite(os.path.join(outputdir, os.path.splitext(fname)[0] + '_l' + '.png'), img_l)
        cv2.imwrite(os.path.join(outputdir, os.path.splitext(fname)[0] + '_r' + '.png'), img_r)

        if fname.startswith('n'):
            img_lu = ProcessImage(rotateImage(org, 45))
            cv2.imwrite(os.path.join(outputdir, os.path.splitext(fname)[0] + '_lu' + '.png'), img_lu)
            img_lu = ProcessImage(rotateImage(org, -45))
            cv2.imwrite(os.path.join(outputdir, os.path.splitext(fname)[0] + '_ru' + '.png'), img_lu)
            
    print('files found:' + str(len(files)))
    
import shutil
def SeparateData(srcdir, traindir, validatedir, testdir, pattern='*'):
    p = Path(srcdir)
    files = list(p.glob(pattern))
    farr = np.asarray(files)
    np.random.shuffle(farr)
    
    length = len(farr)
    train_size = int(length*0.6)
    valid_size = int(length*0.3)
    train, valid, test = farr[:train_size], farr[train_size:train_size+valid_size], farr[train_size+valid_size:]
    
    print('total:{}, train:{}, valid:{}, test:{}'.format(length, len(train), len(valid), len(test)))
    
    if not os.path.exists(traindir):
        os.makedirs(traindir)
    if not os.path.exists(validatedir):
        os.makedirs(validatedir)
    if not os.path.exists(testdir):
        os.makedirs(testdir)
    
    for f in train:
        shutil.copy2(os.path.join(srcdir, f.name), traindir)
    for f in valid:
        shutil.copy2(os.path.join(srcdir, f.name), validatedir)
    for f in test:
        shutil.copy2(os.path.join(srcdir, f.name), testdir)
        
def SeparateSubData(srcdir, desdir, new_len, pattern='*'):
    p = Path(srcdir)
    files = list(p.glob(pattern))
    farr = np.asarray(files)
    np.random.shuffle(farr)
    
    des = farr[:new_len]
    
    if not os.path.exists(desdir):
        os.makedirs(desdir)
    
    for f in des:
        shutil.copy2(os.path.join(srcdir, f.name), desdir)
        
