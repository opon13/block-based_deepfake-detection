# %%
import os
import cv2
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image
from dfx import get_path

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filters', type=str, action='append', default=['-jpegQF90','-jpegQF80','-jpegQF70','-jpegQF60','-jpegQF50', \
                                                                        '-GaussNoise-3', '-GaussNoise-7', '-GaussNoise-15', \
                                                                        '-scaling-50', '-scaling-200', \
                                                                        '-mir-B', '-rot-45', '-rot-135'])
    args = parser.parse_args()

    return args


def Rotation(image_path, path_test, image, fold, rotations: list | None = [45, 135, 225, 315]):
    for rotation in rotations:
        try:
            img = cv2.imread(image_path)
            h, w, _= img.shape
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            rotated_image = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(path_test+"/testing_dset-rot-"+str(rotation)+"/"+fold+"/"+image[:-4]+".png", rotated_image)
        except:
            print("**Errore nell'elaborazione della foto - Rotazione ")


def Mirror(image_path, path_test, image, fold):
    try:
        img = cv2.imread(image_path)
        flipBoth = cv2.flip(img, -1)
        cv2.imwrite(path_test+"/testing_dset-mir-B/"+fold+"/"+image[:-4]+".png", flipBoth)
    except:
        print("**Errore nell'elaborazione della foto - Mirror ")
        

def Scaling(image_path, path_test, image, fold, factors: list | None = [50, 200]):
    for scale_percent in factors:
        try:
            img = cv2.imread(image_path)
            width, height = int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
            cv2.imwrite(path_test+"/testing_dset-scaling-"+str(scale_percent)+"/"+fold+"/"+image[:-4]+".png", resized_image)
        except:
            print("**Errore nell'elaborazione della foto - Zoom")


def GaussNoise(image_path, path_test, image, fold, kernel: list | None =[3,7,9,15]):
    for k in kernel:
        try:
            img = cv2.imread(image_path)
            dst = cv2.GaussianBlur(img,(k,k),cv2.BORDER_DEFAULT)
            cv2.imwrite(path_test+"/testing_dset-GaussNoise-"+str(k)+"/"+fold+"/"+image[:-4]+".png", dst)
        except:
            print("**Errore nell'elaborazione della foto - Gaussian Noice ")


def JPEGCompr(image_path, path_test, image, fold, qfac: list | None = [1,10,20,30,40,50,60,70,80,90]):
    for q in qfac:
        try:
            im = Image.open(image_path)
            im.save(path_test+"/testing_dset-jpegQF"+str(q)+"/"+fold+"/"+image[:-4]+'.jpg',format='jpeg', subsampling=0, quality=q)
        except:
            print("**Errore nell'elaborazione della foto: ", image_path)


def main(parser):
    path_dset = get_path('dataset')
    guidance_path = get_path('guidance')
    path_test = get_path('data_robustness')

    filters = parser.filters

    for fil in filters:
        if not os.path.exists(os.path.join(path_test, f'testing_dset{fil}')):
            os.makedirs(os.path.join(path_test, f'testing_dset{fil}'))
        for architecture in os.listdir(path_dset):
            architecture_path = os.path.join(path_dset, architecture)
            for fold in os.listdir(architecture_path):
                if not os.path.exists(os.path.join(path_test, f'testing_dset{fil}', architecture, fold)):
                    os.makedirs(os.path.join(path_test, f'testing_dset{fil}', architecture, fold))

    guidance_csv = pd.read_csv(guidance_path)
    df = guidance_csv[guidance_csv['label']==2]
    progressive_bar = tqdm(df.iterrows(), total=len(df))
    progressive_bar.desc = 'processing images'
    for _, row in progressive_bar:
        img_path = path_dset+row['image_path']
        img = img_path.split('/')[-1]
        fold = os.path.join(img_path.split('/')[-3], img_path.split('/')[-2])
        JPEGCompr(img_path, path_test, img, fold, qfac=['90','80','70','60','50'])
        GaussNoise(img_path, path_test, img, fold, kernel=[3,7,15])
        Scaling(img_path, path_test, img, fold, factors=[50,200])
        Mirror(img_path, path_test, img, fold)
        Rotation(img_path, path_test, img, fold, rotations=[45,135])

if __name__=='__main__':
    main()
# %%
