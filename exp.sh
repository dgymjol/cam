lrs=(0.001 0.1 0.0001)

num_epoch=100

for lr in ${lrs[@]}
    do

    python main.py --task loc --work-dir work_dir/cbu/resnet50_${lr}_Adam --base-lr ${lr} --num-epoch ${num_epoch} --optimizer Adam --device 0 1 --config ./config/cub/resnet50.yaml
    python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_${lr}_Adam --base-lr ${lr} --num-epoch ${num_epoch} --optimizer Adam --device 0 1 --config ./config/cub/resnet50_cam.yaml

    python main.py --task loc --work-dir work_dir/cbu/resnet50_${lr}_SGD --base-lr ${lr} --num-epoch ${num_epoch} --optimizer SGD --device 0 1 --config ./config/cub/resnet50.yaml
    python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_${lr}_SGD --base-lr ${lr} --num-epoch ${num_epoch} --optimizer SGD --device 0 1 --config ./config/cub/resnet50_cam.yaml

    done


# python main.py --task loc --work-dir work_dir/cbu/resnet50_test_Adam --base-lr 0.01 --num-epoch 1 --optimizer Adam --device 0 1 --config ./config/cub/resnet50.yaml
# python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_test_Adam --base-lr 0.01 --num-epoch 1 --optimizer Adam --device 0 1 --config ./config/cub/resnet50_cam.yaml

# python main.py --task loc --work-dir work_dir/cbu/resnet50_test_SGD --base-lr 0.01 --num-epoch 1 --optimizer SGD --device 0 1 --config ./config/cub/resnet50.yaml
# python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_test_SGD --base-lr 0.01 --num-epoch 1 --optimizer SGD --device 0 1 --config ./config/cub/resnet50_cam.yaml


python main.py --task loc --work-dir work_dir/cub/resnet50_gitconfig --device 0 1 --config ./config/cub/resnet50.yaml
