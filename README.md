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
 $ python main.py --work-dir work_dir/cub/r50cam_0.001 --config ./config/cub/resnet50_cam.yaml --base-lr 0.001 --num-epoch 95
 $ python main.py --work-dir work_dir/cub/r50cam_aug_0.005 --config ./config/cub/resnet50_cam_aug.yaml --base-lr 0.005--num-epoch 95
 ```

### 4. Evaluation (mIoU)
```bash
 $ python evaluation.py --weights ./work_dir/cub/r50cam_0.001:92 --config config/cub/resnet50_cam_eval.yaml
 $ python evaluation.py --weights ./work_dir/cub/r50cam_aug_0.005:67 --config config/cub/resnet50_cam_eval.yaml
 ```

### 5. Visualization for one image
```bash
 $ python visualized_cam.py --weights ./work_dir/cub/r50cam_0.001:92 --image-id 1
 ```

###  6. References : 
- Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A.: Learning deep features for discriminative localization. 
- https://github.com/jiwoon-ahn/irn
- https://github.com/zhoubolei/CAM/tree/c63f2850a7a3dadc21fa1b021875e2d4d053ece5
- https://github.com/zhangyongshun/resnet_finetune_cub
- https://github.com/Uason-Chen/CTR-GCN
