# python run_detect.py --root_path ./datasets/NIPS_Water --data_name NIPS_Water \
# --pretrained_model_path ./pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-270000.pth \
# --pretrain_args_path ./pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
# --seg_len 100 --anomaly_ratio 2 \
# --batch_size 64 --gpu 6 \
# --is_training 0 --IR_mode

python run_detect.py --root_path ./datasets/NIPS_Water --data_name NIPS_Water \
--pretrained_model_path ./pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-270000.pth \
--pretrain_args_path ./pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
--seg_len 100 --anomaly_ratio 2 \
--finetune_layers 1 --dropout 0.2 --neighbor_num 5 --slide_step 1 --learning_rate 1e-5 --train_epochs 1 --tolerance 3 --batch_size 64 --gpu 5 \
--is_training 1