python run_detect.py --root_path ./datasets/PSM/ --data_name PSM \
--pretrained_model_path ./pretrain-library/U2MPSM-Base_dataPSM_dim25_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-490000.pth \
--pretrain_args_path ./pretrain-library/U2MPSM-Base_dataPSM_dim25_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
--seg_len 100 --anomaly_ratio 1 \
--batch_size 32 --gpu 0 \
--is_training 0 --IR_mode

python run_detect.py --root_path ./datasets/PSM/ --data_name PSM \
--pretrained_model_path ./pretrain-library/U2MPSM-Base_dataPSM_dim25_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-490000.pth \
--pretrain_args_path ./pretrain-library/U2MPSM-Base_dataPSM_dim25_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
--seg_len 100 --anomaly_ratio 1 \
--finetune_layers 1 --dropout 0.2 --neighbor_num 10 --slide_step 10 --learning_rate 1e-5 --train_epochs 10 --tolerance 3 --batch_size 32 --gpu 0 \
--is_training 1