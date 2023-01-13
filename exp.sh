python main.py --work-dir work_dir/cub/vgg_gap_bn_gamma_448 --config ./config/cub/vgg_gap_bn.yaml --num-epoch 100 --device 0 1
python main.py --work-dir work_dir/cub/vgg_gap_gamma_448 --config ./config/cub/vgg_gap.yaml --num-epoch 100 --device 0 1

# python evaluation.py --weights work_dir/cub/vgg_gap:33 --device 0 --config config/cub/vgg_gap_eval.yaml
# python evaluation.py --weights work_dir/cub/vgg_gap:31 --device 0 --config config/cub/vgg_gap_eval.yaml


# python evaluation.py --weights work_dir/cub/r50cam_0.001:92 --device 0 1 --config config/cub/resnet50_cam_eval.yaml

# lrs=(0.001 0.005 0.0025 0.001)

# for lr in ${lrs[@]}
#     do
#         python main.py --work-dir work_dir/cub/r50cam_${lr} --config ./config/cub/resnet50_cam.yaml --base-lr ${lr} --num-epoch 95
#         python main.py --work-dir work_dir/cub/r50cam_aug_${lr} --config ./config/cub/resnet50_cam_aug.yaml --base-lr ${lr} --num-epoch 95

#         # python main.py --work-dir work_dir/cub/resnet50_${lr}_adam --config ./config/cub/resnet50_cam.yaml --base-lr ${lr}
#         # python main.py --work-dir work_dir/cub/resnet50_${lr}_pt_adam --config ./config/cub/resnet50_cam.yaml --base-lr ${lr} --weights ./work_dir/cub_good/resnet50_cam:27

#         # python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_adam_rlrp --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml --optimizer Adam --scheduler ReduceLROnPlateau

#         # python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_sgd --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml

#         # python main.py --work-dir work_dir/cub/resnet50_cam_aug_${lr}_sgd --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml
#         # python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_sgd_rlrp --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml --scheduler ReduceLROnPlateau
#         # python main.py --work-dir work_dir/cub/resnet50_cam_aug_${lr}_sgd_rlrp --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml --scheduler ReduceLROnPlateau

#         # python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_adam --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml --optimizer Adam
#         # python main.py --work-dir work_dir/cub/resnet50_cam_aug_${lr}_adam --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml --optimizer Adam
#         # python main.py --work-dir work_disr/cub/resnet50_cam_aug_${lr}_adam_rlrp --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml --optimizer Adam --scheduler ReduceLROnPlateau

#     done

