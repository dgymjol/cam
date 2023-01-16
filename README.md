# CAM based on ResNet50

### 0. Introduction :
- CAM on CUB-200-2011 dataset
- CAM backbone : ResNet50 (pretrained on ImageNet)


### 1.  Virtual Env
conda : 
```bash
 $ bash install.sh 
 ```


### 2. Preprocessing CUB-200-2011 dataset
```bash
 $ cd data/CUB_200_2011
 $ wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
 $ tar -xzvf CUB_200_2011.tgz
 $ python preprocessing.py
```


### 3. Training
```bash
 $ bash exp.sh 
```

### 4. Evaluation
```bash
 $ bash eval.sh
 ```

### 5. Visualization for one image
```bash
 $ python visualized_cam.py --weights work_dir/cub/vgg16_cam_0.001:45 --image-id 1
 ```

###  6. References : 
- Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A.: Learning deep features for discriminative localization. 
- https://github.com/jiwoon-ahn/irn
- https://github.com/zhoubolei/CAM/tree/c63f2850a7a3dadc21fa1b021875e2d4d053ece5
- https://github.com/zhangyongshun/resnet_finetune_cub
- https://github.com/Uason-Chen/CTR-GCN
- https://github.com/won-bae/rethinkingCAM
- https://github.com/junsukchoe/ADL
