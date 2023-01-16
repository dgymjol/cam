# VGG16 - CAM
python main.py --work-dir work_dir/cub/vgg16_cam_0.001 --config ./config/cub/vgg16_cam.yaml 

# ResNet50 - CAM (basic)
python main.py --work-dir work_dir/cub/resnet50_cam_0.01 --config ./config/cub/resnet50_cam.yaml --base-lr 0.01

# ResNet50 - CAM (pytorch model)
python main.py --work-dir work_dir/cub/resnet50_cam_m_0.01 --config ./config/cub/resnet50_cam_m.yaml --base-lr 0.01
