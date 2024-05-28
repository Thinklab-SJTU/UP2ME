for out_len in 50 100 150 200
do
    #immediate forecast
    python run_forecast.py --data_format npy --data_name NIPS_Water --root_path ./datasets/NIPS_Water/ --valid_prop 0.2 \
    --pretrained_model_path pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-500000.pth \
    --pretrain_args_path pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
    --in_len 400 --out_len $out_len \
    --gpu 0 --is_training 0 --IR_mode
    
    #finetune
    python run_forecast.py --data_format npy --data_name NIPS_Water --root_path ./datasets/NIPS_Water/ --valid_prop 0.2 \
    --pretrained_model_path pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-500000.pth \
    --pretrain_args_path pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
    --in_len 400 --out_len $out_len \
    --finetune_layers 1 --neighbor_num 5 --dropout 0.2 --learning_rate 1e-5 --tolerance 3 --slide_step 1 \
    --gpu 0 --is_training 1
done