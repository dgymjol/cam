python evaluation.py --weights work_dir/cub/vgg_gap_256_cam_0.001:45 --device 0 1 --config config/cub/vgg_gap_cam_eval.yaml --results-dir results/cub/new_eval_test


# python main.py --work-dir work_dir/cub/vgg_gap_bn_gamma_448 --config ./config/cub/vgg_gap_bn.yaml --num-epoch 100 --device 0 1
# python main.py --work-dir work_dir/cub/vgg_gap_gamma_448 --config ./config/cub/vgg_gap.yaml --num-epoch 100 --device 0 1

# python evaluation.py --weights work_dir/cub/vgg_gap:33 --device 0 --config config/cub/vgg_gap_eval.yaml
# python evaluation.py --weights work_dir/cub/vgg_gap:31 --device 0 --config config/cub/vgg_gap_eval.yaml


# python evaluation.py --weights work_dir/cub/r50cam_0.001:92 --device 0 1 --config config/cub/resnet50_cam_eval.yaml


# eps=(18)

# for ep in ${eps[@]}
#     do
#         python evaluation.py --weights work_dir/cub/vgg_gap_256_cam_0.0025:${ep} --device 0 1 --config config/cub/vgg_gap_cam_eval.yaml --results-dir results/cub/vgg_gap_cam_0.0025/
#     done


# eps=(11 17)

# for ep in ${eps[@]}
#     do
#         python evaluation.py --weights work_dir/cub/vgg_gap_256_0.001:${ep} --device 0 1 --config config/cub/vgg_gap_eval.yaml --results-dir results/cub/vgg_gap_0.001_wi/
#     done


# eps=(54 154)

# for ep in ${eps[@]}
#     do
#         python evaluation.py --weights work_dir/cub/resnet50_256_0.001:${ep} --device 0 1 --config config/cub/resnet50_eval.yaml --results-dir results/cub/resnet50/
#     done

# eps=(13 32 45)

# for ep in ${eps[@]}
#     do
#         python evaluation.py --weights work_dir/cub/vgg_gap_256_cam_0.001:${ep} --device 0 1 --config config/cub/vgg_gap_cam_eval.yaml --results-dir results/cub/vgg_gap_cam/
#     done

# eps=(58 167)

# for ep in ${eps[@]}
#     do
#         python evaluation.py --weights work_dir/cub/resnet50_256_cam_0.001:${ep} --device 0 1 --config config/cub/resnet50_cam_eval.yaml --results-dir results/cub/vgg_gap_cam/
#     done
