python run_detect.py --root_path ./datasets/SMD/ --data_name SMD \
--pretrained_model_path ./pretrain-library/U2MSMD-Base_dataSMD_dim38_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-140000.pth \
--pretrain_args_path ./pretrain-library/U2MSMD-Base_dataSMD_dim38_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
--seg_len 100 --anomaly_ratio 0.5 \
--batch_size 16 --gpu 0 \
--is_training 0 --IR_mode

python run_detect.py --root_path ./datasets/SMD/ --data_name SMD \
--pretrained_model_path ./pretrain-library/U2MSMD-Base_dataSMD_dim38_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-140000.pth \
--pretrain_args_path ./pretrain-library/U2MSMD-Base_dataSMD_dim38_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
--seg_len 100 --anomaly_ratio 0.5 \
--finetune_layers 1 --dropout 0.2 --neighbor_num 10 --slide_step 10 --learning_rate 1e-5 --train_epochs 10 --tolerance 3 --batch_size 64 --gpu 0 \
--is_training 1