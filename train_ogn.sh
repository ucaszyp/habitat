CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--auto_gpu_config 0 \
-n 4 \
--num_processes_per_gpu 1 \
--num_processes_on_first_gpu 1 \
--sim_gpu_id 1 \
-d saved_hm3d_cerberus/ \
--exp_name exp1 \
--save_periodic 100000
# --print_images 1
# --sem_gt 1 \
# --visualize 1 \