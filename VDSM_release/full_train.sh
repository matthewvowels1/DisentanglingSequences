    python3 main.py --RUN release_test --rnn_layers 3 --rnn_dim 512 --bs 20 --seq_len 20 --epochs 250 --bs_per_epoch 50 \
 --num_test_ids 10 --dataset_name MUG-FED --model_save_interval 50  \
    --train_VDSMSeq False --train_VDSMEncDec True --model_test_interval 10  \
  --anneal_start_dynamics 0.1 --anneal_end_dynamics 0.6 --anneal_frac_dynamics 1  --lr_VDSMEncDec 0.001 --lr_resume 0.0008  --likelihood Bernoulli --z_dim 30 --n_e_w 15 \
  --dynamics_dim 50 --test_temp_id 10  --temp_id_end 10 --temp_id_start 1 --temp_id_frac 2 --anneal_end_id 1 --anneal_start_id 0.1 \
    --anneal_frac_id 3 --anneal_start_t 30.0 --anneal_mid_t1 0.2 --anneal_mid_t2 0.4 --anneal_end_t 1 --anneal_t_midfrac1 0.5 --anneal_t_midfrac2 0.8 \
    --rnn_dropout 0.1


        python3 main.py --RUN release_test --rnn_layers 3 --rnn_dim 512 --bs 20 --seq_len 20 --epochs 200 --bs_per_epoch 50 \
 --num_test_ids 10 --dataset_name MUG-FED --model_save_interval 50 --num_test_ids 10 --pretrained_model_VDSMEncDec 249\
    --train_VDSMSeq True --train_VDSMEncDec False --model_test_interval 10  \
   --anneal_start_dynamics 0.1 --anneal_end_dynamics 1.0 --anneal_frac_dynamics 1   --lr_VDSMSeq 0.001  --likelihood Bernoulli --z_dim 30 --n_e_w 15 \
--dynamics_dim 50 --test_temp_id 10.0 --temp_id_end 10.0 --temp_id_start 10.0 --temp_id_frac 1 --anneal_end_id 1.0 --anneal_start_id .1 \
    --anneal_frac_id 3 --anneal_start_t 0.1 --anneal_mid_t1 0.4 --anneal_mid_t2 0.4 --anneal_end_t 1 --anneal_t_midfrac1 0.5 --anneal_t_midfrac2 0.8 \
    --rnn_dropout 0.1