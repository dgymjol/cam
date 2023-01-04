lrs=(0.01 0.001)



num_epoch=100

for lr in ${lrs[@]}
    do

    python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_${lr}_Adam --base-lr ${lr} --num-epoch ${num_epoch} --optimizer Adam --device 0 1 --config ./config/cub/resnet50.yaml
    python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_${lr}_SGD --base-lr ${lr} --num-epoch ${num_epoch} --optimizer SGD --device 0 1 --config ./config/cifar100/resnet50.yaml

    done

