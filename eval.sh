# VGG16 - CAM evaluation
python evaluation.py --weights work_dir/cub/vgg16_cam_0.001:45 --device 0 1 --config config/cub/vgg16_cam_eval.yaml --results-dir results/cub/new_eval_test

# ResNet50 - CAM (basic) evaluation
python evaluation.py --weights work_dir/cub/resnet50_cam_0.01:32 --device 0 1 --config config/cub/resnet50_cam_eval.yaml --results-dir results/cub/new_eval_test

# ResNet50 - CAM (pytorch model) evaluation
python evaluation.py --weights work_dir/cub/resnet50_cam_m_0.01:20 --device 0 1 --config config/cub/resnet50_cam_m_eval.yaml --results-dir results/cub/new_eval_test
