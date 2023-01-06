lrs=(0.01 0.02 0.005 0.0001)

ep = 1
for lr in ${lrs[@]}
    do
        python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_adam --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml --optimizer Adam --scheduler ReduceLROnPlateau --num-epoch 1
        python main.py --work-dir work_dir/cub/resnet50_cam_aug_${lr}_adam --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml --optimizer Adam --scheduler ReduceLROnPlateau --num-epoch 1

        python main.py --work-dir work_dir/cub/resnet50_cam_${lr}_sgd --base-lr ${lr} --config ./config/cub/resnet50_cam.yaml --num-epoch 1
        python main.py --work-dir work_dir/cub/resnet50_cam_aug_${lr}_sgd --base-lr ${lr} --config ./config/cub/resnet50_cam_aug.yaml --num-epoch 1

    done


python main.py --work-dir work_dir/cub/resnet50 --config ./config/cub/resnet50.yaml --num-epoch 1
python main.py --work-dir work_dir/cub/resnet50_aug --config ./config/cub/resnet50_aug.yaml --num-epoch 1

python main.py --work-dir work_dir/cub/resnet50_0.02 --base-lr 0.02 --config ./config/cub/resnet50.yaml --num-epoch 1
python main.py --work-dir work_dir/cub/resnet50_aug_0.02 --base-lr 0.02 --config ./config/cub/resnet50_aug.yaml --num-epoch 1

python main.py --work-dir work_dir/cub/resnet50_0.005 --base-lr 0.005 --config ./config/cub/resnet50.yaml --num-epoch 1
python main.py --work-dir work_dir/cub/resnet50_aug_0.005 --base-lr 0.005 --config ./config/cub/resnet50_aug.yaml --num-epoch 1

python main.py --work-dir work_dir/cub/resnet50_cam --config ./config/cub/resnet50_cam.yaml --num-epoch 1
python main.py --work-dir work_dir/cub/resnet50_cam_aug --config ./config/cub/resnet50_cam_aug.yaml --num-epoch 1



