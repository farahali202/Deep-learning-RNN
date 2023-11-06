# Build a Deep Face Detection Model 

# 1-setup and get data

# 1-2 collect image usong openCv


```python
import os
import time
import uuid
import cv2
```


```python
image_path=os.path.join('Dat','images')
number_images=30
```


```python
cap=cv2.VideoCapture(0)
for img in range(number_images):
    print('collecting image {}'.format(img))
    ret,frame=cap.read()
    imgname=os.path.join(image_path,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname,frame)
    cv2.imshow('frame',frame)
    time.sleep(0.5)

    if cv2.waitKey(0) & 0xFF ==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
```

# 1-3 annotate images with labelme


```python
!labelme
```

    [INFO   ] __init__:get_config:70 - Loading config file from: C:\Users\Farah\.labelmerc
    

# 2-review dataset and build image loading function


```python
import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
```

# 2-1 load image into TF data pipeline


```python
images=tf.data.Dataset.list_files('Dat\\images\\*.jpg',shuffle=False)
```


```python
images.as_numpy_iterator().next()
```




    b'Dat\\images\\2e696fce-791d-11ee-b239-c8d3ffe44550.jpg'




```python
def load_image(x):
    byt_img=tf.io.read_file(x)
    img=tf.io.decode_jpeg(byt_img)
    return img
```


```python
images=images.map(load_image)
```


```python
images.as_numpy_iterator().next()
```




    array([[[104, 124, 125],
            [102, 122, 123],
            [104, 122, 126],
            ...,
            [ 71,  83,  81],
            [ 68,  82,  82],
            [ 69,  83,  83]],
    
           [[103, 123, 124],
            [102, 122, 123],
            [104, 122, 124],
            ...,
            [ 71,  83,  81],
            [ 70,  82,  82],
            [ 68,  82,  82]],
    
           [[103, 123, 122],
            [103, 123, 122],
            [104, 124, 123],
            ...,
            [ 73,  84,  80],
            [ 72,  82,  81],
            [ 70,  82,  80]],
    
           ...,
    
           [[ 69,  80,  74],
            [ 69,  80,  74],
            [ 69,  82,  75],
            ...,
            [ 34,  26,  23],
            [ 34,  26,  23],
            [ 34,  26,  23]],
    
           [[ 70,  81,  77],
            [ 69,  80,  76],
            [ 69,  81,  77],
            ...,
            [ 34,  25,  20],
            [ 35,  26,  21],
            [ 35,  26,  21]],
    
           [[ 73,  84,  80],
            [ 71,  82,  78],
            [ 69,  81,  79],
            ...,
            [ 34,  25,  20],
            [ 36,  25,  21],
            [ 36,  25,  21]]], dtype=uint8)



# 2-2 view raw images with matplotlib


```python
img_generator=images.batch(4).as_numpy_iterator()
```


```python
plot_images=img_generator.next()
```


```python
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx,img in enumerate(plot_images):
    ax[idx].imshow(img)
plt.show()
```


    
![png](output_19_0.png)
    


# 3-partition unaugmented Data

# 3-1-manually splitt data into train test val


```python
60*.7# 42 to train
```




    42.0




```python
60*.15 #9 to test and 9 for val
```




    9.0



# 3-2move the matching labels


```python
for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('Dat',folder,'images')):
        filename=file.split('.')[0]+'.json'
        existing_filepath=os.path.join('Dat','labels',filename)
        if os.path.exists(existing_filepath):
            print(existing_filepath)
            new_filepath=os.path.join('Dat',folder,'labels',filename)
            print(new_filepath)
            os.replace(existing_filepath,new_filepath)
```

    Dat\labels\2e696fce-791d-11ee-b239-c8d3ffe44550.json
    Dat\train\labels\2e696fce-791d-11ee-b239-c8d3ffe44550.json
    Dat\labels\3fba6958-791d-11ee-98e9-c8d3ffe44550.json
    Dat\train\labels\3fba6958-791d-11ee-98e9-c8d3ffe44550.json
    Dat\labels\6d7b0b1b-791d-11ee-8be0-c8d3ffe44550.json
    Dat\train\labels\6d7b0b1b-791d-11ee-8be0-c8d3ffe44550.json
    Dat\labels\8935d1ce-791d-11ee-aab1-c8d3ffe44550.json
    Dat\train\labels\8935d1ce-791d-11ee-aab1-c8d3ffe44550.json
    Dat\labels\8c76c94c-791d-11ee-9d69-c8d3ffe44550.json
    Dat\train\labels\8c76c94c-791d-11ee-9d69-c8d3ffe44550.json
    Dat\labels\8eb5aeed-791d-11ee-81e9-c8d3ffe44550.json
    Dat\train\labels\8eb5aeed-791d-11ee-81e9-c8d3ffe44550.json
    Dat\labels\9022769f-791d-11ee-9dfc-c8d3ffe44550.json
    Dat\train\labels\9022769f-791d-11ee-9dfc-c8d3ffe44550.json
    Dat\labels\90e78413-791d-11ee-b0e9-c8d3ffe44550.json
    Dat\train\labels\90e78413-791d-11ee-b0e9-c8d3ffe44550.json
    Dat\labels\91d9b8b5-791d-11ee-82de-c8d3ffe44550.json
    Dat\train\labels\91d9b8b5-791d-11ee-82de-c8d3ffe44550.json
    Dat\labels\9312ab01-791d-11ee-a27b-c8d3ffe44550.json
    Dat\train\labels\9312ab01-791d-11ee-a27b-c8d3ffe44550.json
    Dat\labels\949587c8-791d-11ee-8c46-c8d3ffe44550.json
    Dat\train\labels\949587c8-791d-11ee-8c46-c8d3ffe44550.json
    Dat\labels\95a3bbe8-791d-11ee-936b-c8d3ffe44550.json
    Dat\train\labels\95a3bbe8-791d-11ee-936b-c8d3ffe44550.json
    Dat\labels\9a29e1e2-791d-11ee-81d3-c8d3ffe44550.json
    Dat\train\labels\9a29e1e2-791d-11ee-81d3-c8d3ffe44550.json
    Dat\labels\9b12ecde-791d-11ee-8223-c8d3ffe44550.json
    Dat\train\labels\9b12ecde-791d-11ee-8223-c8d3ffe44550.json
    Dat\labels\9d8da1b4-791d-11ee-9a23-c8d3ffe44550.json
    Dat\train\labels\9d8da1b4-791d-11ee-9a23-c8d3ffe44550.json
    Dat\labels\9ed4af84-791d-11ee-b230-c8d3ffe44550.json
    Dat\train\labels\9ed4af84-791d-11ee-b230-c8d3ffe44550.json
    Dat\labels\a00b145c-791d-11ee-bdf8-c8d3ffe44550.json
    Dat\train\labels\a00b145c-791d-11ee-bdf8-c8d3ffe44550.json
    Dat\labels\a12aa2fd-791d-11ee-8b85-c8d3ffe44550.json
    Dat\train\labels\a12aa2fd-791d-11ee-8b85-c8d3ffe44550.json
    Dat\labels\a44c9981-791d-11ee-9e49-c8d3ffe44550.json
    Dat\train\labels\a44c9981-791d-11ee-9e49-c8d3ffe44550.json
    Dat\labels\a5aaac87-791d-11ee-a363-c8d3ffe44550.json
    Dat\train\labels\a5aaac87-791d-11ee-a363-c8d3ffe44550.json
    Dat\labels\a6d9dc88-791d-11ee-bbf4-c8d3ffe44550.json
    Dat\train\labels\a6d9dc88-791d-11ee-bbf4-c8d3ffe44550.json
    Dat\labels\a8c448a9-791d-11ee-80da-c8d3ffe44550.json
    Dat\train\labels\a8c448a9-791d-11ee-80da-c8d3ffe44550.json
    Dat\labels\b48fd3e1-791d-11ee-8915-c8d3ffe44550.json
    Dat\train\labels\b48fd3e1-791d-11ee-8915-c8d3ffe44550.json
    Dat\labels\b5620806-791d-11ee-97ae-c8d3ffe44550.json
    Dat\train\labels\b5620806-791d-11ee-97ae-c8d3ffe44550.json
    Dat\labels\b74c513a-791d-11ee-a3aa-c8d3ffe44550.json
    Dat\train\labels\b74c513a-791d-11ee-a3aa-c8d3ffe44550.json
    Dat\labels\b91e0367-791d-11ee-892e-c8d3ffe44550.json
    Dat\train\labels\b91e0367-791d-11ee-892e-c8d3ffe44550.json
    Dat\labels\c0131cf9-791d-11ee-9098-c8d3ffe44550.json
    Dat\train\labels\c0131cf9-791d-11ee-9098-c8d3ffe44550.json
    Dat\labels\c136c052-791d-11ee-a179-c8d3ffe44550.json
    Dat\train\labels\c136c052-791d-11ee-a179-c8d3ffe44550.json
    Dat\labels\c24079d6-791d-11ee-b7ac-c8d3ffe44550.json
    Dat\train\labels\c24079d6-791d-11ee-b7ac-c8d3ffe44550.json
    Dat\labels\c44a4910-791d-11ee-8190-c8d3ffe44550.json
    Dat\train\labels\c44a4910-791d-11ee-8190-c8d3ffe44550.json
    Dat\labels\c5d701c0-791d-11ee-8e86-c8d3ffe44550.json
    Dat\train\labels\c5d701c0-791d-11ee-8e86-c8d3ffe44550.json
    Dat\labels\c8cf4607-791d-11ee-82ba-c8d3ffe44550.json
    Dat\train\labels\c8cf4607-791d-11ee-82ba-c8d3ffe44550.json
    Dat\labels\cad2dc64-791d-11ee-8387-c8d3ffe44550.json
    Dat\train\labels\cad2dc64-791d-11ee-8387-c8d3ffe44550.json
    Dat\labels\ccf8a0ef-791d-11ee-9699-c8d3ffe44550.json
    Dat\train\labels\ccf8a0ef-791d-11ee-9699-c8d3ffe44550.json
    Dat\labels\cef87a24-791d-11ee-98e4-c8d3ffe44550.json
    Dat\train\labels\cef87a24-791d-11ee-98e4-c8d3ffe44550.json
    Dat\labels\d0373e0b-791d-11ee-bf9d-c8d3ffe44550.json
    Dat\train\labels\d0373e0b-791d-11ee-bf9d-c8d3ffe44550.json
    Dat\labels\d1e9a22e-791d-11ee-9814-c8d3ffe44550.json
    Dat\train\labels\d1e9a22e-791d-11ee-9814-c8d3ffe44550.json
    Dat\labels\d2ef8438-791d-11ee-98b0-c8d3ffe44550.json
    Dat\train\labels\d2ef8438-791d-11ee-98b0-c8d3ffe44550.json
    Dat\labels\d3c5dd6e-791d-11ee-9209-c8d3ffe44550.json
    Dat\train\labels\d3c5dd6e-791d-11ee-9209-c8d3ffe44550.json
    Dat\labels\d5ba43ef-791d-11ee-8123-c8d3ffe44550.json
    Dat\train\labels\d5ba43ef-791d-11ee-8123-c8d3ffe44550.json
    Dat\labels\d65ea181-791d-11ee-807d-c8d3ffe44550.json
    Dat\train\labels\d65ea181-791d-11ee-807d-c8d3ffe44550.json
    Dat\labels\d7093018-791d-11ee-99f1-c8d3ffe44550.json
    Dat\train\labels\d7093018-791d-11ee-99f1-c8d3ffe44550.json
    Dat\labels\8b16ae6b-791d-11ee-bf51-c8d3ffe44550.json
    Dat\test\labels\8b16ae6b-791d-11ee-bf51-c8d3ffe44550.json
    Dat\labels\8da7378d-791d-11ee-91fc-c8d3ffe44550.json
    Dat\test\labels\8da7378d-791d-11ee-91fc-c8d3ffe44550.json
    Dat\labels\96dea672-791d-11ee-b924-c8d3ffe44550.json
    Dat\test\labels\96dea672-791d-11ee-b924-c8d3ffe44550.json
    Dat\labels\a3421a61-791d-11ee-8774-c8d3ffe44550.json
    Dat\test\labels\a3421a61-791d-11ee-8774-c8d3ffe44550.json
    Dat\labels\a7e16cc7-791d-11ee-a85f-c8d3ffe44550.json
    Dat\test\labels\a7e16cc7-791d-11ee-a85f-c8d3ffe44550.json
    Dat\labels\b833cfa9-791d-11ee-948b-c8d3ffe44550.json
    Dat\test\labels\b833cfa9-791d-11ee-948b-c8d3ffe44550.json
    Dat\labels\ba4bd052-791d-11ee-9cf8-c8d3ffe44550.json
    Dat\test\labels\ba4bd052-791d-11ee-9cf8-c8d3ffe44550.json
    Dat\labels\c33b0319-791d-11ee-8558-c8d3ffe44550.json
    Dat\test\labels\c33b0319-791d-11ee-8558-c8d3ffe44550.json
    Dat\labels\cbea2e0f-791d-11ee-a5fb-c8d3ffe44550.json
    Dat\test\labels\cbea2e0f-791d-11ee-a5fb-c8d3ffe44550.json
    Dat\labels\d0fd643b-791d-11ee-b522-c8d3ffe44550.json
    Dat\test\labels\d0fd643b-791d-11ee-b522-c8d3ffe44550.json
    Dat\labels\9c6b3b32-791d-11ee-862f-c8d3ffe44550.json
    Dat\val\labels\9c6b3b32-791d-11ee-862f-c8d3ffe44550.json
    Dat\labels\a23cbc43-791d-11ee-9e9c-c8d3ffe44550.json
    Dat\val\labels\a23cbc43-791d-11ee-9e9c-c8d3ffe44550.json
    Dat\labels\a9bc5cf5-791d-11ee-b2c7-c8d3ffe44550.json
    Dat\val\labels\a9bc5cf5-791d-11ee-b2c7-c8d3ffe44550.json
    Dat\labels\b66cb1aa-791d-11ee-a422-c8d3ffe44550.json
    Dat\val\labels\b66cb1aa-791d-11ee-a422-c8d3ffe44550.json
    Dat\labels\c700986a-791d-11ee-ba1b-c8d3ffe44550.json
    Dat\val\labels\c700986a-791d-11ee-ba1b-c8d3ffe44550.json
    Dat\labels\c7e1f883-791d-11ee-900c-c8d3ffe44550.json
    Dat\val\labels\c7e1f883-791d-11ee-900c-c8d3ffe44550.json
    Dat\labels\ce03962a-791d-11ee-8c20-c8d3ffe44550.json
    Dat\val\labels\ce03962a-791d-11ee-8c20-c8d3ffe44550.json
    Dat\labels\d4b348b9-791d-11ee-810a-c8d3ffe44550.json
    Dat\val\labels\d4b348b9-791d-11ee-810a-c8d3ffe44550.json
    

# 4-aply image augmentation on images and labels using Albumentations

# 4-1 setup albumentations transform pipline


```python
import albumentations as alb
```


```python
augmentor= alb.Compose([alb.RandomCrop(width=450,height=450),
                        alb.HorizontalFlip(p=0.5),
                        alb.RandomBrightnessContrast(p=0.2),
                        alb.RandomGamma(p=0.2),
                        alb.RGBShift(p=0.2),
                        alb.VerticalFlip(p=0.5)],
                       bbox_params=alb.BboxParams(format='albumentations',
                                                 label_fields=['class_labels'])
                      )
```

# 4-2 load a test image and annotation with opencv and json


```python
img=cv2.imread(os.path.join('Dat','train','images','3fba6958-791d-11ee-98e9-c8d3ffe44550.jpg'))
```


```python
img.shape
```




    (480, 640, 3)




```python
with open(os.path.join('Dat','train','labels','3fba6958-791d-11ee-98e9-c8d3ffe44550.json'),'r') as f:
    label=json.load(f)
```


```python
label['shapes']
```




    [{'label': 'face',
      'points': [[217.6190476190476, 126.50793650793648],
       [500.1587301587301, 447.1428571428571]],
      'group_id': None,
      'description': '',
      'shape_type': 'rectangle',
      'flags': {}}]



# 4-3 extract coordinats and rescale to match image resolution


```python
coords=[0,0,0,0]
coords[0]=label['shapes'][0]['points'][0][0]
coords[1]=label['shapes'][0]['points'][0][1]
coords[2]=label['shapes'][0]['points'][1][0]
coords[3]=label['shapes'][0]['points'][1][1]
```


```python
coords
```




    [217.6190476190476, 126.50793650793648, 500.1587301587301, 447.1428571428571]




```python
coords=list(np.divide(coords,[640,480,640,480]))
```


```python
coords
```




    [0.34002976190476186, 0.263558201058201, 0.7814980158730158, 0.931547619047619]



# 4.4 apply augmenattions and view results


```python
augmented=augmentor(image=img,bboxes=[coords],class_labels=['face'])
```


```python
type(augmented)
```




    dict




```python
augmented.keys()
```




    dict_keys(['image', 'bboxes', 'class_labels'])




```python
augmented['image'].shape
```




    (450, 450, 3)




```python
augmented['bboxes']
```




    [(0.19915343915343908,
      0.057460317460317545,
      0.8270194003527336,
      0.7699823633156967)]




```python
#visualising
cv2.rectangle(augmented['image'],
    tuple(np.multiply(augmented['bboxes'][0][:2],[450,450]).astype(int)),#top corner
    tuple(np.multiply(augmented['bboxes'][0][2:],[450,450]).astype(int)),#bottom corner
              (255,255,0),2)#blue green red,tickness)#[450,450] for rescaling
plt.imshow(augmented['image'])
```




    <matplotlib.image.AxesImage at 0x179e0adb700>




    
![png](output_46_1.png)
    



```python
augmented['bboxes'][0][:2]
```




    (0.19915343915343908, 0.057460317460317545)




```python
augmented['bboxes'][0][2:]
```




    (0.8270194003527336, 0.7699823633156967)



# 5-build and run augmentation pipeline

# 5-1 run augmentation pipeline


```python
for folder in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join('Dat', folder, 'images')):  
        img = cv2.imread(os.path.join('Dat', folder, 'images', image))

        coords = [0, 0, 0.0001, 0.0001]
        label_path = os.path.join('Dat', folder, 'labels', f'{image.split(".")[0]}.json')

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640, 480, 640, 480]))

        try:
            for x in range(60):  # creating 60 augmented images for every single img
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', folder, 'images', f'{image.split(".")[0]}.{x}.jpg'),
                            augmented['image'])

                annotation = {}
                annotation['images'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bboxes'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                    else:
                        annotation['bboxes'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bboxes'] = [0, 0, 0, 0]
                    annotation['class'] = 0

                with open(os.path.join('aug_data', folder, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)
        except Exception as e:
            print(e)

```

# 5-2 Load augmented images to tensorflow dataset


```python
train_img=tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg',shuffle=False)
train_img=train_img.map(load_image)
train_img=train_img.map(lambda x:tf.image.resize(x,(120,120)))
train_img=train_img.map(lambda x:x/255)
```


```python
test_img=tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg',shuffle=False)
test_img=test_img.map(load_image)
test_img=test_img.map(lambda x:tf.image.resize(x,(120,120)))
test_img=test_img.map(lambda x:x/255)
```


```python
val_img=tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg',shuffle=False)
val_img=val_img.map(load_image)
val_img=val_img.map(lambda x:tf.image.resize(x,(120,120)))
val_img=val_img.map(lambda x:x/255)
```


```python
train_img.as_numpy_iterator().next()
```




    array([[[0.452451  , 0.4877451 , 0.4759804 ],
            [0.46023285, 0.49019608, 0.48020834],
            [0.47107843, 0.4867647 , 0.48284313],
            ...,
            [0.39197305, 0.40226716, 0.37775734],
            [0.40784314, 0.41384804, 0.39148283],
            [0.3995711 , 0.4083946 , 0.37751225]],
    
           [[0.46213236, 0.49803922, 0.48596814],
            [0.47058824, 0.49411765, 0.4862745 ],
            [0.46317402, 0.4971814 , 0.4858456 ],
            ...,
            [0.40925246, 0.4170956 , 0.39846814],
            [0.4053309 , 0.41317403, 0.40055147],
            [0.39577207, 0.4057598 , 0.3822304 ]],
    
           [[0.45784312, 0.49558824, 0.49166667],
            [0.46617648, 0.4897059 , 0.4897059 ],
            [0.45196077, 0.4971201 , 0.49019608],
            ...,
            [0.3970588 , 0.40490195, 0.39313725],
            [0.40392157, 0.4117647 , 0.39227942],
            [0.39332107, 0.4011642 , 0.38645834]],
    
           ...,
    
           [[0.30147058, 0.31715685, 0.31323528],
            [0.29656863, 0.32156864, 0.30539215],
            [0.28584558, 0.3211397 , 0.30153185],
            ...,
            [0.15196079, 0.15196079, 0.12058824],
            [0.17107843, 0.15245098, 0.12843138],
            [0.16813725, 0.15061274, 0.11973039]],
    
           [[0.29246324, 0.3153799 , 0.30784315],
            [0.29705882, 0.31715685, 0.30147058],
            [0.29797795, 0.31807598, 0.3023897 ],
            ...,
            [0.13872549, 0.14166667, 0.11862745],
            [0.1538603 , 0.13523284, 0.11807598],
            [0.16452205, 0.1413603 , 0.11783088]],
    
           [[0.28866422, 0.31954658, 0.3043505 ],
            [0.2908701 , 0.31207108, 0.2959559 ],
            [0.30618873, 0.31403187, 0.2973652 ],
            ...,
            [0.12873775, 0.13523284, 0.11286765],
            [0.13639706, 0.12990196, 0.11029412],
            [0.14515932, 0.12922794, 0.11170343]]], dtype=float32)



# 6-prepare labels:

# 6-1 build label and loading function


```python
def load_labels(label_path):
    with open(label_path.numpy(),'r',encoding="utf-8") as f:
        label=json.load(f)
    
    return [label['class']],label['bboxes']
```

# 6-2 load labels to tensorflow dataset


```python
train_labels=tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json',shuffle=False)
train_labels=train_labels.map(lambda x:tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))
```


```python
test_labels=tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json',shuffle=False)
test_labels=test_labels.map(lambda x:tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))
```


```python
# Create a dataset of file paths for validation labels
val_labels=tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json',shuffle=False)
# Map a function to load labels using tf.py_function
val_labels=val_labels.map(lambda x:tf.py_function(load_labels,[x],[tf.uint8,tf.float16]))
```


```python
train_labels.as_numpy_iterator().next()
```




    (array([1], dtype=uint8),
     array([0.329 , 0.2725, 0.922 , 0.9355], dtype=float16))



# 7-combine label and image sample

# 7-1-check partition lengths


```python
len(train_img),len(test_img),len(val_img),len(train_labels),len(test_labels),len(val_labels)
```




    (2520, 600, 600, 2520, 600, 600)



# 7-2 create final datasets (images /labels)


```python
train=tf.data.Dataset.zip((train_img,train_labels))
train=train.shuffle(5000)
train=train.batch(8)
train=train.prefetch(4)
```


```python
test=tf.data.Dataset.zip((test_img,test_labels))
test=test.shuffle(5000)
test=test.batch(8)
test=test.prefetch(4)
```


```python
val=tf.data.Dataset.zip((val_img,val_labels))
val=val.shuffle(5000)
val=val.batch(8)
val=val.prefetch(4)
```

# 7-3 view images and annotations


```python
data_samp=train.as_numpy_iterator()
```


```python
res=data_samp.next()
```


```python
res[1]
```




    (array([[1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]], dtype=uint8),
     array([[0.000e+00, 9.808e-02, 3.335e-01, 8.882e-01],
            [3.833e-01, 4.226e-01, 9.478e-01, 1.000e+00],
            [3.503e-01, 1.797e-01, 8.901e-01, 9.204e-01],
            [1.836e-01, 5.862e-02, 7.866e-01, 9.121e-01],
            [1.611e-01, 0.000e+00, 7.256e-01, 5.908e-01],
            [9.247e-02, 3.179e-01, 7.275e-01, 1.000e+00],
            [0.000e+00, 0.000e+00, 5.347e-01, 6.543e-01],
            [3.184e-01, 2.470e-04, 9.253e-01, 6.880e-01]], dtype=float16))




```python
res[0][7]
```




    array([[[0.76911765, 0.84362745, 0.82009804],
            [0.7738358 , 0.85128677, 0.8262868 ],
            [0.77892154, 0.8485294 , 0.825     ],
            ...,
            [0.7757966 , 0.79540443, 0.771875  ],
            [0.7712622 , 0.7908701 , 0.76734066],
            [0.7721814 , 0.7887255 , 0.75876224]],
    
           [[0.7631127 , 0.8401961 , 0.8153799 ],
            [0.7872549 , 0.86960787, 0.8421569 ],
            [0.76960784, 0.8519608 , 0.8245098 ],
            ...,
            [0.7882353 , 0.80784315, 0.78431374],
            [0.7834559 , 0.80306375, 0.7756128 ],
            [0.7733456 , 0.79295343, 0.76795346]],
    
           [[0.7919118 , 0.86151963, 0.8379902 ],
            [0.7831495 , 0.85765934, 0.8341299 ],
            [0.7724265 , 0.8469363 , 0.8234069 ],
            ...,
            [0.7921569 , 0.8117647 , 0.78431374],
            [0.7873775 , 0.81041664, 0.7726716 ],
            [0.7941176 , 0.8150123 , 0.7840074 ]],
    
           ...,
    
           [[0.7563726 , 0.80196077, 0.7911765 ],
            [0.7504289 , 0.7728554 , 0.7645221 ],
            [0.6903799 , 0.66047794, 0.6477941 ],
            ...,
            [0.55557597, 0.3805147 , 0.34381127],
            [0.6179534 , 0.3336397 , 0.29454657],
            [0.6819853 , 0.37297794, 0.35545343]],
    
           [[0.73131126, 0.7563113 , 0.7411152 ],
            [0.71727943, 0.71979165, 0.7001838 ],
            [0.6587623 , 0.58817405, 0.56164217],
            ...,
            [0.6292892 , 0.38308823, 0.3550858 ],
            [0.68878675, 0.3897059 , 0.34822303],
            [0.6884191 , 0.37769607, 0.32702205]],
    
           [[0.6474265 , 0.65232843, 0.6420343 ],
            [0.62463236, 0.615625  , 0.60343134],
            [0.6369485 , 0.53498775, 0.50802696],
            ...,
            [0.69197303, 0.36862746, 0.31746325],
            [0.68927693, 0.37493873, 0.33406863],
            [0.64007354, 0.37487745, 0.32463235]]], dtype=float32)




```python
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx in range(4):
    sample_img=res[0][idx]
    sample_coord=res[1][1][idx]
    
    cv2.rectangle(sample_img,
                 tuple(np.multiply(sample_coord[:2],[120,120]).astype(int)),
                 tuple(np.multiply(sample_coord[2:],[120,120]).astype(int)),
                 (255,0,0),2)
    ax[idx].imshow(sample_img)
plt.show()
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![png](output_77_1.png)
    


# 8-Build deep learning using functional API

# 8-1-import layers and base network


```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input, Conv2D, MaxPooling2D, Flatten, Add, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
```

# 8.2 download VGG16


```python
vgg=VGG16(include_top=False)
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58889256/58889256 [==============================] - 9s 0us/step
    


```python
vgg.summary()
```

    Model: "vgg16"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, None, None, 3)]   0         
                                                                     
     block1_conv1 (Conv2D)       (None, None, None, 64)    1792      
                                                                     
     block1_conv2 (Conv2D)       (None, None, None, 64)    36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, None, None, 64)    0         
                                                                     
     block2_conv1 (Conv2D)       (None, None, None, 128)   73856     
                                                                     
     block2_conv2 (Conv2D)       (None, None, None, 128)   147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, None, None, 128)   0         
                                                                     
     block3_conv1 (Conv2D)       (None, None, None, 256)   295168    
                                                                     
     block3_conv2 (Conv2D)       (None, None, None, 256)   590080    
                                                                     
     block3_conv3 (Conv2D)       (None, None, None, 256)   590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, None, None, 256)   0         
                                                                     
     block4_conv1 (Conv2D)       (None, None, None, 512)   1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, None, None, 512)   0         
                                                                     
     block5_conv1 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, None, None, 512)   2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, None, None, 512)   0         
                                                                     
    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
    


```python
def build_model():
    input_layer=Input(shape=(120,120,3))#120 pixel
    vgg=VGG16(include_top=False)(input_layer)
    
    #classification model
    f1=GlobalMaxPooling2D()(vgg)
    class1=Dense(2048,activation='relu')(f1)
    class2=Dense(1,activation='sigmoid')(class1)
    
    #Bounding box mmodel
    f2=GlobalMaxPooling2D()(vgg)
    regress1=Dense(2048,activation='relu')(f2)
    regress2=Dense(4,activation='sigmoid')(regress1)
    
    facetracker=Model(inputs=input_layer,outputs=[class2,regress2])
    return facetracker
```

# 8-4 test out neurall network


```python
facetracker=build_model()
```


```python
facetracker.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_4 (InputLayer)           [(None, 120, 120, 3  0           []                               
                                    )]                                                                
                                                                                                      
     vgg16 (Functional)             (None, None, None,   14714688    ['input_4[0][0]']                
                                    512)                                                              
                                                                                                      
     global_max_pooling2d_1 (Global  (None, 512)         0           ['vgg16[0][0]']                  
     MaxPooling2D)                                                                                    
                                                                                                      
     global_max_pooling2d_2 (Global  (None, 512)         0           ['vgg16[0][0]']                  
     MaxPooling2D)                                                                                    
                                                                                                      
     dense (Dense)                  (None, 2048)         1050624     ['global_max_pooling2d_1[0][0]'] 
                                                                                                      
     dense_2 (Dense)                (None, 2048)         1050624     ['global_max_pooling2d_2[0][0]'] 
                                                                                                      
     dense_1 (Dense)                (None, 1)            2049        ['dense[0][0]']                  
                                                                                                      
     dense_3 (Dense)                (None, 4)            8196        ['dense_2[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 16,826,181
    Trainable params: 16,826,181
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
X,y=train.as_numpy_iterator().next()
```


```python
X.shape
```




    (8, 120, 120, 3)




```python
classes,coords=facetracker.predict(X)
```

    1/1 [==============================] - 2s 2s/step
    


```python
classes,coords
```




    (array([[0.4311141 ],
            [0.41676328],
            [0.41702825],
            [0.41832733],
            [0.4308431 ],
            [0.46467805],
            [0.4643247 ],
            [0.48946378]], dtype=float32),
     array([[0.5403711 , 0.5194771 , 0.6118205 , 0.6021487 ],
            [0.44778574, 0.49181154, 0.6389432 , 0.58344924],
            [0.5229158 , 0.53238535, 0.56080425, 0.57235473],
            [0.5264521 , 0.5230922 , 0.6492294 , 0.6087199 ],
            [0.4750103 , 0.4639694 , 0.59384024, 0.639464  ],
            [0.43089926, 0.41914192, 0.5042732 , 0.54811984],
            [0.41945887, 0.5119362 , 0.53090936, 0.50395656],
            [0.47861448, 0.45963386, 0.62820077, 0.51228416]], dtype=float32))




```python
#we need a specific loss function fot classification and other fot regression
```

# 9-Define Losses and Optimizers

# 9-1 Define Optimizer and LR


```python
#calculate learning decay
len(train)
```




    315




```python
#specify learning decay
batches_per_epoch=315
lr_decay=(1./0.75-1)/batches_per_epoch
```


```python
#how much learning rate is going to deop each time throigh one particular epoch
lr_decay
```




    0.001058201058201058




```python
initial_learning_rate = 0.0001
lr_decay = 0.001058201058201058
decay_steps = 10000

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=decay_steps, decay_rate=lr_decay, staircase=True
)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

# 9-2 create localization loss and classification loss


```python
def localization_loss(y_true,y_hat):
    delta_coord=tf.reduce_sum(tf.square(y_true[:,:2]-y_hat[:,:2]))
    
    h_true=y_true[:,3]-y_true[:,1]#height
    w_true=y_true[:,2]-y_true[:,0]#width
    
    h_pred=y_hat[:,3]-y_hat[:,1]#height
    w_pred=y_hat[:,2]-y_hat[:,0]#width
    
    delta_size=tf.reduce_sum(tf.square(w_true-w_pred)+tf.square(h_true-h_pred))
    
    return delta_size+delta_coord
```


```python
classloss=tf.keras.losses.BinaryCrossentropy()
regloss=localization_loss
```

# 9-3 Test out Loss metrics


```python
y
```




    (array([[1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]], dtype=uint8),
     array([[0.     , 0.01563, 0.4983 , 0.721  ],
            [0.1611 , 0.     , 0.7256 , 0.591  ],
            [0.4653 , 0.6494 , 1.     , 1.     ],
            [0.2898 , 0.1857 , 0.879  , 0.9614 ],
            [0.     , 0.4517 , 0.258  , 0.9985 ],
            [0.0947 , 0.32   , 0.7295 , 1.     ],
            [0.2496 , 0.2144 , 0.835  , 0.955  ],
            [0.083  , 0.3137 , 0.559  , 0.931  ]], dtype=float16))




```python
localization_loss(y[1],coords).numpy()
```




    5.682553




```python
classloss(y[0],classes)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.81909347>




```python
classloss(y[0],classes).numpy()
```




    0.81909347




```python
regloss(y[1],coords).numpy() 
```




    5.682553



# 10-Train neural network

# 10 -1 create pipeline(custom model class)


```python
class FaceTracker(Model):
    def __init__(self,eyetracker,**kwargs):
        super().__init__(**kwargs)
        self.model=eyetracker
    def compile(self,opt,classloss,localizationloss,**kwargs):
        super().compile(**kwargs)
        self.closs=classloss
        self.lloss=localizationloss
        self.opt=opt
    def train_step(self,batch,**kwargs):
        X,y=batch
        
        with tf.GradientTape() as tape:
            classes,coords=self.model(X,training=True)
            batch_classloss=self.closs(y[0],classes)
            batch_localizationloss=self.lloss(tf.cast(y[1],tf.float32),coords)
            
            total_loss=batch_localizationloss+0.5*batch_classloss
            
            grad=tape.gradient(total_loss,self.model.trainable_variables)
            
        opt.apply_gradients(zip(grad,self.model.trainable_variables))
        
        return {"total loss" :total_loss,"class_loss":batch_classloss,"regressor_loss":batch_localizationloss}
    
    def test_step(self,batch,**kwargs):
        X,y=batch
        classes,coords=self.model(X,training=False)
        
        batch_classloss=self.closs(y[0],classes)
        batch_localizationloss=self.lloss(tf.cast(y[1],tf.float32),coords)
            
        total_loss=batch_localizationloss+0.5*batch_classloss
        
        return {"total loss" :total_loss,"class_loss":batch_classloss,"regressor_loss":batch_localizationloss}
    def call(self,X,**kwargs):
        return self.model(X,**kwargs)
```


```python
model=FaceTracker(facetracker)
```


```python
model.compile(opt,classloss,regloss)
```

# 10.2 train


```python
logdir='logs'
```


```python
tensorflowboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
```


```python
hist=model.fit(train,epochs=10,validation_data=val,
              callbacks=[tensorflowboard_callback])
```

    Epoch 1/10
    315/315 [==============================] - 1151s 4s/step - total loss: 0.0094 - class_loss: 2.4300e-06 - regressor_loss: 0.0094 - val_total loss: 0.0787 - val_class_loss: 1.3411e-07 - val_regressor_loss: 0.0787
    Epoch 2/10
    315/315 [==============================] - 1209s 4s/step - total loss: 0.0094 - class_loss: 2.0070e-06 - regressor_loss: 0.0094 - val_total loss: 0.3804 - val_class_loss: 0.0173 - val_regressor_loss: 0.3718
    Epoch 3/10
    315/315 [==============================] - 976s 3s/step - total loss: 0.0071 - class_loss: 1.1819e-06 - regressor_loss: 0.0071 - val_total loss: 0.6856 - val_class_loss: 0.1004 - val_regressor_loss: 0.6354
    Epoch 4/10
    315/315 [==============================] - 968s 3s/step - total loss: 0.0086 - class_loss: 1.4786e-06 - regressor_loss: 0.0086 - val_total loss: 0.1013 - val_class_loss: 1.1548e-06 - val_regressor_loss: 0.1013
    Epoch 5/10
    315/315 [==============================] - 946s 3s/step - total loss: 0.0085 - class_loss: 1.1496e-06 - regressor_loss: 0.0085 - val_total loss: 0.2239 - val_class_loss: 0.0068 - val_regressor_loss: 0.2205
    Epoch 6/10
    315/315 [==============================] - 1311s 4s/step - total loss: 0.0089 - class_loss: 3.0290e-05 - regressor_loss: 0.0089 - val_total loss: 0.2082 - val_class_loss: 0.0126 - val_regressor_loss: 0.2019
    Epoch 7/10
    315/315 [==============================] - 1513s 5s/step - total loss: 0.1362 - class_loss: 0.0141 - regressor_loss: 0.1292 - val_total loss: 2.9648 - val_class_loss: 1.3284 - val_regressor_loss: 2.3006
    Epoch 8/10
    315/315 [==============================] - 1632s 5s/step - total loss: 0.0240 - class_loss: 4.1054e-04 - regressor_loss: 0.0238 - val_total loss: 0.1263 - val_class_loss: 0.0011 - val_regressor_loss: 0.1257
    Epoch 9/10
    315/315 [==============================] - 1589s 5s/step - total loss: 0.0076 - class_loss: 2.7962e-05 - regressor_loss: 0.0076 - val_total loss: 0.3734 - val_class_loss: 0.2188 - val_regressor_loss: 0.2640
    Epoch 10/10
    315/315 [==============================] - 1424s 5s/step - total loss: 0.0056 - class_loss: 1.4700e-05 - regressor_loss: 0.0056 - val_total loss: 0.0679 - val_class_loss: 0.0020 - val_regressor_loss: 0.0669
    
callbacks=[tensorflowboard_callback]: A list of callback functions. Callbacks are functions that are executed at specific points during training. In this case, the tensorflowboard_callback is specified, which suggests the use of a TensorBoard callback for visualization of the training process.

If you have TensorBoard set up and configured correctly, this callback (tensorflowboard_callback) would log information about the training process to TensorBoard, allowing you to visualize metrics, model architecture, and more. Make sure that tensorflowboard_callback is an instance of tf.keras.callbacks.TensorBoard and that TensorBoard is properly configured in your environment.call(self, X, **kwargs):: This method allows the model to be called as a function. It delegates the call to the underlying eye tracker model (self.model), passing the input X and any additional keyword arguments.

```python
hist.history
```




    {'total loss': [0.004398586228489876,
      0.011361195705831051,
      0.002611022675409913,
      0.01003931276500225,
      0.005229824222624302,
      0.046487677842378616,
      0.06532993912696838,
      0.010238641873002052,
      0.0052671972662210464,
      0.002932129893451929],
     'class_loss': [1.8477496723789955e-06,
      4.023318069812376e-07,
      7.599601872243511e-07,
      5.736951038670668e-07,
      9.760289003679645e-07,
      1.4901162970204496e-08,
      5.616134876618162e-05,
      3.904120603692718e-06,
      1.8402979549136944e-06,
      1.4826705410087015e-06],
     'regressor_loss': [0.004397662356495857,
      0.011360994540154934,
      0.00261064269579947,
      0.010039025917649269,
      0.005229336209595203,
      0.04648767039179802,
      0.06530185788869858,
      0.010236689820885658,
      0.005266277119517326,
      0.002931388560682535],
     'val_total loss': [0.07866649329662323,
      0.38039717078208923,
      0.6856470704078674,
      0.10134024918079376,
      0.22385813295841217,
      0.20819643139839172,
      2.964759111404419,
      0.1262812465429306,
      0.373371422290802,
      0.06790180504322052],
     'val_class_loss': [1.3411052179890248e-07,
      0.01727576181292534,
      0.10043029487133026,
      1.1548427210072987e-06,
      0.006753823719918728,
      0.012629223056137562,
      1.3283753395080566,
      0.0011416313936933875,
      0.21878160536289215,
      0.0019864095374941826],
     'val_regressor_loss': [0.07866642624139786,
      0.371759295463562,
      0.6354319453239441,
      0.1013396680355072,
      0.22048121690750122,
      0.20188182592391968,
      2.3005714416503906,
      0.12571042776107788,
      0.2639806270599365,
      0.06690859794616699]}



# 10.3 plot performance


```python
fig,ax=plt.subplots(ncols=3,figsize=(20,5))

ax[0].plot(hist.history['total loss'],color='red',label='loss')
ax[0].plot(hist.history['val_total loss'],color='orange',label='val loss')
ax[0].title.set_text('loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'],color='red',label='class loss')
ax[1].plot(hist.history['val_class_loss'],color='orange',label='val class loss')
ax[1].title.set_text('classification loss')
ax[1].legend()

ax[2].plot(hist.history['regressor_loss'],color='red',label='regress loss')
ax[2].plot(hist.history['val_regressor_loss'],color='orange',label='val regress loss')
ax[2].title.set_text('regression loss')
ax[2].legend()

plt.show()
```


    
![png](output_121_0.png)
    


# 11 Make predictions

# 11-1 Make predictions on test set


```python
test_data=test.as_numpy_iterator()
```


```python
test_sample=test_data.next()
```


```python
yhat=facetracker.predict(test_sample[0])
```

    1/1 [==============================] - 1s 1s/step
    


```python
test_sample[0].shape
```




    (8, 120, 120, 3)




```python
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx in range(4):
    sample_img=test_sample[0][idx]
    sample_coord=yhat[1][idx]
    
    if yhat[0][idx]>0.5:
        cv2.rectangle(sample_img,
                      tuple(np.multiply(sample_coord[:2],[120,120]).astype(int)),
                      tuple(np.multiply(sample_coord[2:],[120,120]).astype(int)),
                    (255,0,0),2 )
    ax[idx].imshow(sample_img)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![png](output_128_1.png)
    


# 11-2 save the model


```python
from tensorflow.keras.models import load_model
```


```python
facetracker.save('facetracker.h5')
```

    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    


```python
facetracker=load_model('facetracker.h5')
```

    WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
    

# 11.3 real time detection


```python
cap=cv2.VideoCapture(0)
while cap.isOpened():
    _,frame=cap.read()
    frame=frame[50:500,50:500,:]
    
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    resized=tf.image.resize(rgb,(120,120))
    
    yhat=facetracker.predict(np.expand_dims(resized/255,0))
    sample_coors=yhat[1][0]
    
    if yhat[0]>0.5:
        #controls the main rectagle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coors[:2],[450,450]).astype(int)),
                      tuple(np.multiply(sample_coors[2:],[450,450]).astype(int)),
                      (255,0,0),2
                     )
        #controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coors[:2],[450,450]).astype(int),
                                   [0,-30] )),
                      tuple(np.add(np.multiply(sample_coors[:2],[450,450]).astype(int),
                                   [80,0] )),
                      (255,0,0),-1
                     )
        #controls the text rendred
        cv2.putText(frame,'face',tuple(np.add(np.multiply(sample_coors[:2],[450,450]).astype(int),
                                   [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('EyeTrack',frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
```

    1/1 [==============================] - 0s 339ms/step
    1/1 [==============================] - 0s 163ms/step
    1/1 [==============================] - 0s 151ms/step
    1/1 [==============================] - 0s 155ms/step
    1/1 [==============================] - 0s 142ms/step
    1/1 [==============================] - 0s 146ms/step
    1/1 [==============================] - 0s 145ms/step
    1/1 [==============================] - 0s 144ms/step
    1/1 [==============================] - 0s 132ms/step
    1/1 [==============================] - 0s 124ms/step
    1/1 [==============================] - 0s 143ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 166ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 138ms/step
    1/1 [==============================] - 0s 153ms/step
    1/1 [==============================] - 0s 145ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 125ms/step
    1/1 [==============================] - 0s 128ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 164ms/step
    1/1 [==============================] - 0s 148ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 149ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 135ms/step
    1/1 [==============================] - 0s 131ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 141ms/step
    1/1 [==============================] - 0s 128ms/step
    1/1 [==============================] - 0s 124ms/step
    1/1 [==============================] - 0s 128ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 128ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 125ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 130ms/step
    1/1 [==============================] - 0s 133ms/step
    1/1 [==============================] - 0s 143ms/step
    1/1 [==============================] - 0s 130ms/step
    1/1 [==============================] - 0s 161ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 138ms/step
    1/1 [==============================] - 0s 135ms/step
    1/1 [==============================] - 0s 173ms/step
    1/1 [==============================] - 0s 128ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 128ms/step
    1/1 [==============================] - 0s 155ms/step
    1/1 [==============================] - 0s 138ms/step
    1/1 [==============================] - 0s 123ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 130ms/step
    1/1 [==============================] - 0s 140ms/step
    1/1 [==============================] - 0s 140ms/step
    1/1 [==============================] - 0s 133ms/step
    1/1 [==============================] - 0s 143ms/step
    1/1 [==============================] - 0s 137ms/step
    1/1 [==============================] - 0s 143ms/step
    1/1 [==============================] - 0s 140ms/step
    1/1 [==============================] - 0s 135ms/step
    1/1 [==============================] - 0s 137ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 147ms/step
    1/1 [==============================] - 0s 132ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 135ms/step
    1/1 [==============================] - 0s 135ms/step
    1/1 [==============================] - 0s 140ms/step
    1/1 [==============================] - 0s 145ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 151ms/step
    1/1 [==============================] - 0s 147ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 147ms/step
    1/1 [==============================] - 0s 132ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 142ms/step
    1/1 [==============================] - 0s 140ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 144ms/step
    1/1 [==============================] - 0s 125ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 125ms/step
    1/1 [==============================] - 0s 148ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 133ms/step
    1/1 [==============================] - 0s 128ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 142ms/step
    1/1 [==============================] - 0s 133ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 124ms/step
    1/1 [==============================] - 0s 148ms/step
    1/1 [==============================] - 0s 125ms/step
    1/1 [==============================] - 0s 150ms/step
    1/1 [==============================] - 0s 132ms/step
    1/1 [==============================] - 0s 125ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 124ms/step
    1/1 [==============================] - 0s 133ms/step
    1/1 [==============================] - 0s 133ms/step
    1/1 [==============================] - 0s 124ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 132ms/step
    1/1 [==============================] - 0s 157ms/step
    1/1 [==============================] - 0s 151ms/step
    1/1 [==============================] - 0s 169ms/step
    1/1 [==============================] - 0s 125ms/step
    1/1 [==============================] - 0s 151ms/step
    1/1 [==============================] - 0s 135ms/step
    1/1 [==============================] - 0s 124ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 134ms/step
    1/1 [==============================] - 0s 125ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 125ms/step
    1/1 [==============================] - 0s 141ms/step
    1/1 [==============================] - 0s 145ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 146ms/step
    1/1 [==============================] - 0s 150ms/step
    1/1 [==============================] - 0s 145ms/step
    1/1 [==============================] - 0s 143ms/step
    1/1 [==============================] - 0s 145ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 134ms/step
    1/1 [==============================] - 0s 150ms/step
    1/1 [==============================] - 0s 137ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 138ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 134ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 137ms/step
    1/1 [==============================] - 0s 163ms/step
    1/1 [==============================] - 0s 151ms/step
    1/1 [==============================] - 0s 140ms/step
    1/1 [==============================] - 0s 135ms/step
    1/1 [==============================] - 0s 137ms/step
    1/1 [==============================] - 0s 142ms/step
    1/1 [==============================] - 0s 133ms/step
    1/1 [==============================] - 0s 138ms/step
    1/1 [==============================] - 0s 132ms/step
    1/1 [==============================] - 0s 147ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 132ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 131ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 142ms/step
    1/1 [==============================] - 0s 153ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 137ms/step
    1/1 [==============================] - 0s 124ms/step
    1/1 [==============================] - 0s 158ms/step
    1/1 [==============================] - 0s 131ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 143ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 124ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 130ms/step
    1/1 [==============================] - 0s 132ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 131ms/step
    1/1 [==============================] - 0s 129ms/step
    1/1 [==============================] - 0s 128ms/step
    1/1 [==============================] - 0s 132ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 131ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 172ms/step
    1/1 [==============================] - 0s 140ms/step
    1/1 [==============================] - 0s 127ms/step
    1/1 [==============================] - 0s 126ms/step
    1/1 [==============================] - 0s 133ms/step
    1/1 [==============================] - 0s 151ms/step
    


```python

```
