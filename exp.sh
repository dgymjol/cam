lrs=(0.001 0.0001 0.00001)
num_epoch=100

for lr in ${lrs[@]}
    do

    python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_${lr}_Adam --base-lr ${lr} --num-epoch ${num_epoch} --optimizer Adam --device 0 1 --config ./config/cifar100/resnet50_cam.yaml
    python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_${lr}_SGD --base-lr ${lr} --num-epoch ${num_epoch} --optimizer SGD --device 0 1 --config ./config/cifar100/resnet50_cam.yaml

    done

python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_lr_0.01 --base-lr 0.01 --num-epoch 100 --optimizer SGD --device 0 1 --config ./config/cifar100/resnet50_cam.yaml