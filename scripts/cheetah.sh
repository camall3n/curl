CUDA_VISIBLE_DEVICES=5 python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --action_repeat 4 \
    --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp/cheetah \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 2e-4 --actor_lr 1e-4 --eval_freq 1000 --batch_size 128 --num_train_steps 500000
