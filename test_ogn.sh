CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --split train --eval 1 --auto_gpu_config 0 --sim_gpu_id 1 \
-n 1 --num_eval_episodes 1 --max_episode_length 500 \
--num_processes_per_gpu 1 --num_processes_on_first_gpu 1 \
--print_images 1 --random 0 --save_np 0 --get_bird 1 \
--load /mnt/yupeng/OGN/Object-Goal-Navigation/pretrained_models/sem_exp.pth
# CUDA_VISIBLE_DEVICES=2 python main.py --split val_mini --eval 1 --auto_gpu_config 0 \
# -n 2 --num_eval_episodes 30 --num_processes_on_first_gpu 2 \
# --load /mnt/yupeng/OGN/Object-Goal-Navigation/pretrained_models/sem_exp.pth
# --visualize 1 --player 1
# --print_images 1 -d results_1/ --exp_name exp_debug