 lrs=(0.1 0.0125)
 num_epoch=200

 for lr in ${lrs[@]}
     do

     python main.py --work-dir work_dir/cifar100/resnet50_${lr}_Adam --base-lr ${lr} --num-epoch ${num_epoch} --optimizer Adam --device 0 1
     python main.py --work-dir work_dir/cifar100/resnet50_${lr}_SGD --base-lr ${lr} --num-epoch ${num_epoch} --optimizer SGD --device 0 1

     done

python main.py --task loc --work-dir work_dir/cbu/resnet50_cam_test --base-lr 0.001 --num-epoch 1 --optimizer Adam --device 0 1 --config ./config/cifar100/resnet50_cam.yaml