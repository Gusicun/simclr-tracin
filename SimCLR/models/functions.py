import os
import numpy as np
import pandas as pd
import cv2
import base64
from pathlib import Path
from PIL import Image
import requests
import torch
from deepface.detectors import FaceDetector
from torch import nn
from PIL import Image
import cv2 as cv
from torchvision.transforms import transforms
import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image

#--------------------------------------------------

def initialize_input(img1_path, img2_path = None):

    if type(img1_path) == list:
        bulkProcess = True
        img_list = img1_path.copy()
    else:
        bulkProcess = False

        if (
                (type(img2_path) == str and img2_path != None) #exact image path, base64 image
                or (isinstance(img2_path, np.ndarray) and img2_path.any()) #numpy array
        ):
            img_list = [[img1_path, img2_path]]
        else: #analyze function passes just img1_path
            img_list = [img1_path]

    return img_list, bulkProcess

def initialize_folder():
    home = get_deepface_home()

    if not os.path.exists(home+"/.deepface"):
        os.makedirs(home+"/.deepface")
        print("Directory ", home, "/.deepface created")

    if not os.path.exists(home+"/.deepface/weights"):
        os.makedirs(home+"/.deepface/weights")
        print("Directory ", home, "/.deepface/weights created")

def get_deepface_home():
    return str(os.getenv('DEEPFACE_HOME', default=Path.home()))

def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def load_image(img):
    exact_image = False; base64_img = False; url_img = False


    r_path="D:/DeepFace/demo-master/back/"
    img=r_path+img
    if type(img).__module__ == np.__name__:
        exact_image = True

    elif len(img) > 11 and img[0:11] == "data:image/":
        base64_img = True

    elif len(img) > 11 and img.startswith("http"):
        url_img = True

    #---------------------------

    if base64_img == True:
        img = loadBase64Img(img)

    elif url_img:
        img = np.array(Image.open(requests.get(img, stream=True).raw))

    elif exact_image != True: #image path passed as input
        if os.path.isfile(img) != True:
            raise ValueError("Confirm that ",img," exists")

        img = cv2.imread(img)

    return img

def detect_face(img, detector_backend = 'opencv', grayscale = False, enforce_detection = True, align = True):

    img_region = [0, 0, img.shape[0], img.shape[1]]

    #----------------------------------------------
    #people would like to skip detection and alignment if they already have pre-processed images
    if detector_backend == 'skip':
        return img, img_region

    #----------------------------------------------

    #detector stored in a global variable in FaceDetector object.
    #this call should be completed very fast because it will return found in memory
    #it will not build face detector model in each call (consider for loops)
    face_detector = FaceDetector.build_model(detector_backend)

    try:
        detected_face, img_region = FaceDetector.detect_face(face_detector, detector_backend, img, align)
    except: #if detected face shape is (0, 0) and alignment cannot be performed, this block will be run
        detected_face = None

    if (isinstance(detected_face, np.ndarray)):
        return detected_face, img_region
    else:
        if detected_face == None:
            if enforce_detection != True:
                return img, img_region
            else:
                raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

def normalize_input(img, normalization = 'base'):

    #issue 131 declares that some normalization techniques improves the accuracy

    if normalization == 'base':
        return img
    else:
        #@trevorgribble and @davedgd contributed this feature

        img *= 255 #restore input in scale of [0, 255] because it was normalized in scale of  [0, 1] in preprocess_face

        if normalization == 'raw':
            pass #return just restored pixels

        elif normalization == 'Facenet':
            mean, std = img.mean(), img.std()
            img = (img - mean) / std

        elif(normalization=="Facenet2018"):
            # simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
            img /= 127.5
            img -= 1

        elif normalization == 'VGGFace':
            # mean subtraction based on VGGFace1 training data
            img[..., 0] -= 93.5940
            img[..., 1] -= 104.7624
            img[..., 2] -= 129.1863

        elif(normalization == 'VGGFace2'):
            # mean subtraction based on VGGFace2 training data
            img[..., 0] -= 91.4953
            img[..., 1] -= 103.8827
            img[..., 2] -= 131.0912

        elif(normalization == 'ArcFace'):
            #Reference study: The faces are cropped and resized to 112×112,
            #and each pixel (ranged between [0, 255]) in RGB images is normalised
            #by subtracting 127.5 then divided by 128.
            img -= 127.5
            img /= 128

    #-----------------------------

    return img
def SaltAndPepper(src,percetage):
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randR=np.random.randint(0,src.shape[0]-1)
        randG=np.random.randint(0,src.shape[1]-1)
        randB=np.random.randint(0,3)
        if np.random.randint(0,1)==0:
            SP_NoiseImg[randR,randG,randB]=0
        else:
            SP_NoiseImg[randR,randG,randB]=255
    return SP_NoiseImg
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = True,data_arg=True,arg=args):
    #img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    from PIL import ImageEnhance
    img = load_image(img)
    base_img = img.copy()
    img, region = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection, align = align)
    #cv.imwrite(r'D:\DeepFace\demo-master\back\deepface-m\tests\data_crop_align\img'+str(i+1)+'.jpg', img)

    ##simclr式预处理
    if data_arg:
        #s=1
        #color_jitter = transforms.ColorJitter(0.5 * s, 0 * s, 0 * s, 0 * s)
        print('this img has been argued')
        data_arg=transforms.Compose([#transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(5, resample=False, expand=False, center=None),
            #transforms.RandomApply([color_jitter], p=0.8),
            transforms.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0),#hue有限制，在【-0.5,0.5】
            #transforms.RandomGrayscale(p=0.2),
            #GaussianBlur(kernel_size=int(0.1 * 224))
        ])
        #img = Image.fromarray(img)
        #img = data_arg(img)
        #img = cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR)
        #基本参数
        brightness = 0.9
        sharpness = 3.0
        color = 1.5
        contrast = 1.5
        #开始处理
        enh_bri = ImageEnhance.Brightness(img)
        enh_col = ImageEnhance.Color(img)
        enh_sha = ImageEnhance.Sharpness(img)
        enh_con = ImageEnhance.Contrast(img)
        #图像处理
        img = enh_bri.enhance(brightness)
    #img = enh_col.enhance(color)
    #img = enh_sha.enhance(sharpness)
    #img = enh_con.enhance(contrast)

    else:
        print('this picture is org')

    #到此为止是新添加的
    #base_img = img.copy()
    #如果使用对齐和crop，应该去掉这一句，同时原来region没用上，应该有一步裁剪就对了
    #img, region = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection, align = align)
    #print(region)
    #--------------------------

    if img.shape[0] == 0 or img.shape[1] == 0:
        if enforce_detection == True:
            raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
        else: #restore base image
            img = base_img.copy()

    #--------------------------

    #post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #---------------------------------------------------
    #resize image to expected shape

    # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    #------------------------------------------

    #double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    #---------------------------------------------------

    #normalizing the image pixels

    img_pixels = image.img_to_array(img) #what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]

    #---------------------------------------------------

    if return_region == True:
        return img_pixels, region
    else:
        return img_pixels

def find_input_shape(model):

    #face recognition models have different size of inputs
    #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.

    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    #----------------------
    #issue 289: it seems that tf 2.5 expects you to resize images with (x, y)
    #whereas its older versions expect (y, x)

    if tf_major_version == 2 and tf_minor_version >= 5:
        x = input_shape[0]; y = input_shape[1]
        input_shape = (y, x)

    #----------------------

    if type(input_shape) == list: #issue 197: some people got array here instead of tuple
        input_shape = tuple(input_shape)

    return input_shape
