# MLDL 2021 Course
## BiseNet
Starting code for the student belonging to the project "Real-time domain adaptation in semantic segmentation" <br>
BiSeNet based on pytorch 0.4.1 and python 3.6

## Group 1 Fork
Currently working at point 3

## Dataset  
Download CamVid dataset from [Google Drive](https://drive.google.com/file/d/1CKtkLRVU4tGbqLSyFEtJMoZV2ZZ2KDeA/view?usp=sharing) 
Download IDDA dataset from [Google Drive](https://drive.google.com/file/d/1GiUjXp1YBvnJjAf1un07hdHFUrchARa0/view)
Note: classes_info.json file needs to be modified by changing the first couple of brakets '[]' to {} and deleting the last comma.
  
## Train
```
python train.py
```  

## Adversarial Train
```
python train_v2.py
```  

## Test
```
python test.py
```
