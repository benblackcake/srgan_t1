import cv2
import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
import random

# Get the Image
def imread(path):
    img = cv2.imread(path)
    return img

def imsave(image, path, config):
    #checkimage(image)
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))

    # NOTE: because normial, we need mutlify 255 back    
    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)
    print("imshow")
    #checkimage(image)
    if __DEBUG__SHOW__IMAGE :
        imBGR2RGB = cv2.cvtColor(image.astype(np.float32),cv2.COLOR_BGR2RGB)
        plt.imshow(imBGR2RGB)
        plt.show()

    
def checkimage(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)

def downSample(image,scale=3):
    h,w,_ =image.shape
    h_n = h//scale
    w_n = w//scale
    img = np.full((h_n,w_n,_),0)
    
    for i in range(0,h):
        for j in range(0,w):
            if(i % scale==0 and j % scale==0):
                img[i//scale][j//scale] = image[i][j]
    return img



def shuffle_files(filenames):
    return random.shuffle(filenames)


def randomCrop(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img
    
def crop_center(img,cropx,cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def process_sub_image(file_name, img_size=96, random_crop=False):
    img = imread(file_name)
    h,w,_ = img.shape
    if h < img_size or w < img_size: 
        raise
    if random_crop:
        return randomCrop(img,img_size,img_size)

    else:
        return crop_center(img,img_size,img_size)

def fft_batch(batch_img):
    pass


def get_files_list(args):
    """
    return file list
    """

    train_filenames = np.array(glob.glob(os.path.join(args.train_dir, '**', '*.*'), recursive=True))
    val_filenames = np.array(glob.glob(os.path.join('Benchmarks', '**', '*_HR.png'), recursive=True))
    eval_indices = np.random.randint(len(train_filenames), size=len(val_filenames))
    eval_filenames = train_filenames[eval_indices[:119]]
    
    random.shuffle(train_filenames)
    random.shuffle(val_filenames)
    random.shuffle(eval_filenames)
    
    print(train_filenames)
    print(val_filenames)
    print(eval_filenames)
    return train_filenames, val_filenames, eval_filenames

