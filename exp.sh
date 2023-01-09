# python main.py --work-dir work_dir/cub/resnet --config ./config/cub/resnet.yaml --base-lr 0.1
# python main.py --work-dir work_dir/cub/resnet50_${lr}_pt_sgd --config ./config/cub/resnet50_cam.yaml --base-lr ${lr} --weights ./work_dir/cub_good/resnet50_cam:27
# python evaluation.py --weights ./work_dir/cub/resnet50_cam_plz:45 --device 0 1 --config config/cub/resnet50_cam_eval.yaml


lrs=(0.005 0.0025 0.001)

for lr in ${lrs[@]}
    do
        python main.py --work-dir work_dir/cub/resnet50_${lr} --config ./config/cub/resnet50_cam.yaml --base-lr ${lr} --num-epoch 1
        python main.py --work-dir work_dir/cub/resnet50_${lr}_pt --config ./config/cub/resnet50_cam.yaml --base-lr ${lr} --weights ./work_dir/cub_good/resnet50_cam:27 --num-epoch 1

        python main.py --work-dir work_dir/cub/resnet50_aug_${lr} --config ./config/cub/resnet50_cam_aug.yaml --base-lr ${lr} --num-epoch 1
        python main.py --work-dir work_dir/cub/resnet50_aug_${lr}_pt --config ./config/cub/resnet50_aug_cam.yaml --base-lr ${lr} --weights ./work_dir/cub_good/resnet50_cam:27 --num-epoch 1
        # python main.py --work-dir work_dir/cub/resnet50_${lr}_adam --config ./config/cub/resnet50_cam.yaml --base-lr ${lr}
        # python main.py --work-dir work_dir/cub/resnet50_${lr}_pt_adam --config ./config/cub/resnet50_cam.yaml --base-lr ${lr} --weights ./work_dir/cub_good/resnet50_cam:27

        # python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_adam_rlrp --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml --optimizer Adam --scheduler ReduceLROnPlateau

        # python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_sgd --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml

        # python main.py --work-dir work_dir/cub/resnet50_cam_aug_${lr}_sgd --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml
        # python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_sgd_rlrp --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml --scheduler ReduceLROnPlateau
        # python main.py --work-dir work_dir/cub/resnet50_cam_aug_${lr}_sgd_rlrp --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml --scheduler ReduceLROnPlateau

        # python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_adam --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml --optimizer Adam
        # python main.py --work-dir work_dir/cub/resnet50_cam_aug_${lr}_adam --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml --optimizer Adam
        # python main.py --work-dir work_disr/cub/resnet50_cam_aug_${lr}_adam_rlrp --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml --optimizer Adam --scheduler ReduceLROnPlateau

    done


# python main.py --work-dir work_dir/cub/resnet50_test_ --config ./config/cub/resnet50.yaml --base-lr 0.005
# python main.py --work-dir work_dir/cub/resnet50_aug_transfer --config ./config/cub/resnet50_aug.yaml --weights work_dir/cub/resnet50:62 --base-lr 0.0001

# python main.py --work-dir work_dir/cub/resnet50_0.02 --base-lr 0.02 --config ./config/cub/resnet50.yaml
# python main.py --work-dir work_dir/cub/resnet50_aug_0.02 --base-lr 0.02 --config ./config/cub/resnet50_aug.yaml

# python main.py --work-dir work_dir/cub/resnet50_0.005 --base-lr 0.005 --config ./config/cub/resnet50.yaml
# python main.py --work-dir work_dir/cub/resnet50_aug_0.005 --base-lr 0.005 --config ./config/cub/resnet50_aug.yaml

# python main.py --work-dir work_dir/cub/resnet50_cam --config ./config/cub/resnet50_cam.yaml
# python main.py --work-dir work_dir/cub/resnet50_cam_aug --config ./config/cub/resnet50_cam_aug.yaml



# python main.py --work-dir work_dir/cub/resnet50_cam --base-lr 0.005 --config ./config/cub/resnet50.yaml --scheduler ReduceLROnPlateau
# python main.py --work-dir work_dir/cub/resnet50_cam_aug --base-lr 0.005 --config ./config/cub/resnet50_aug.yaml --scheduler ReduceLROnPlateau


