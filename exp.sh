 lrs=(0.0001 0.0005 0.001 )
 num_epoch=200

 for lr in ${lrs[@]}
     do

     python main.py --work-dir work_dir/cifar100/resnet50_${lr}_Adam --base-lr ${lr} --num-epoch ${num_epoch} --optimizer Adam --device 0 1
     python main.py --work-dir work_dir/cifar100/resnet50_${lr}_SGD --base-lr ${lr} --num-epoch ${num_epoch} --optimizer SGD --device 0 1

     done