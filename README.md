# MLDL 2021 Course
## BiseNet
Starting code for the student belonging to the project "Real-time domain adaptation in semantic segmentation" <br>
BiSeNet based on pytorch 0.4.1 and python 3.6

## Group 1 Fork -- Final Version

## Dataset  
Download CamVid dataset from [Google Drive](https://drive.google.com/file/d/1CKtkLRVU4tGbqLSyFEtJMoZV2ZZ2KDeA/view?usp=sharing) <br>
Download IDDA dataset from [Google Drive](https://drive.google.com/file/d/1GiUjXp1YBvnJjAf1un07hdHFUrchARa0/view) <br>
Note: classes_info.json file needs to be modified by changing the first couple of brakets '[]' to {} and deleting the last comma.
  
## Train
```
python segmentation_train.py
```  

## Adversarial Train
```
python adversarial_train.py
```  

## Test
```
python test.py
```
