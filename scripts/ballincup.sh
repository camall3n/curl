CUDA_VISIBLE_DEVICES=5 python train.py \
    --domain_name ball_in_cup \
    --task_name catch \
    --encoder_type pixel \
    --action_repeat 4 \
    --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp/ball_in_cup \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-4 --eval_freq 1000 --batch_size 128 --num_train_steps 500000
