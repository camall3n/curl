onager prelaunch +jobname curl-bic +command "unbuffer python train.py --domain_name ball_in_cup --task_name catch --encoder_type pixel --action_repeat 4 --pre_transform_image_size 100 --image_size 84 --work_dir ./tmp/ball_in_cup --agent curl_sac --frame_stack 3 --critic_lr 1e-3 --actor_lr 1e-4 --eval_freq 1000 --batch_size 128 --num_train_steps 500000" +arg --seed {1..20}

onager prelaunch +jobname curl-cartpole +command "unbuffer python train.py --domain_name cartpole --task_name swingup --encoder_type pixel --_repeat 8 --pre_transform_image_size 100 --image_size 84 --work_dir ./tmp/cartpole --agent curl_sac --frame_stack 3 --critic_lr 1e-3 --actor_lr 1e-4 --eval_freq 1000 --batch_size 128 --num_train_steps 500000" +arg --seed {1..20}

onager prelaunch +jobname curl-cheetah +command "unbuffer python train.py --domain_name cheetah --task_name run --encoder_type pixel --action_repeat 4 --pre_transform_image_size 100 --image_size 84 --work_dir ./tmp/cheetah --agent curl_sac --frame_stack 3 --critic_lr 2e-4 --actor_lr 1e-4 --eval_freq 1000 --batch_size 128 --num_train_steps 500000" +arg --seed {1..20}

onager prelaunch +jobname curl-finger +command "unbuffer python train.py --domain_name finger --task_name spin --encoder_type pixel --action_repeat 2 --pre_transform_image_size 100 --image_size 84 --work_dir ./tmp/finger --agent curl_sac --frame_stack 3 --critic_lr 1e-3 --actor_lr 1e-4 --eval_freq 1000 --batch_size 128 --num_train_steps 500000" +arg --seed {1..20}

onager prelaunch +jobname curl-reacher +command "unbuffer python train.py --domain_name reacher --task_name easy --encoder_type pixel --action_repeat 4 --pre_transform_image_size 100 --image_size 84 --work_dir ./tmp/reacher --agent curl_sac --frame_stack 3 --critic_lr 1e-3 --actor_lr 1e-4 --eval_freq 1000 --batch_size 128 --num_train_steps 500000" +arg --seed {1..20}

onager prelaunch +jobname curl-walk +command "unbuffer python train.py --domain_name walker --task_name walk --encoder_type pixel --action_repeat 2 --pre_transform_image_size 100 --image_size 84 --work_dir ./tmp/walker --agent curl_sac --frame_stack 3 --critic_lr 1e-3 --actor_lr 1e-4 --eval_freq 1000 --batch_size 128 --num_train_steps 500000" +arg --seed {1..20}

