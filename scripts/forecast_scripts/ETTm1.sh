for out_len in 96 192 336 720
do
    #immediate reaction forecast
    # python run_forecast.py --data_format csv --data_name ETTm1 --root_path ./datasets/ETT/ --data_path ETTm1.csv \
    # --data_split 34560,11520,11520 --checkpoints ./forecast_checkpoints/ \
    # --pretrained_model_path pretrain-library/U2METTm1-Base_dataETTm1_dim7_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-30000.pth \
    # --pretrain_args_path pretrain-library/U2METTm1-Base_dataETTm1_dim7_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
    # --in_len 336 --out_len $out_len \
    # --gpu 0 --is_training 0 --IR_mode

    #finetune
    python run_forecast.py --data_format csv --data_name ETTm1 --root_path ./datasets/ETT/ --data_path ETTm1.csv \
    --data_split 34560,11520,11520 --checkpoints ./forecast_checkpoints/ \
    --pretrained_model_path pretrain-library/U2METTm1-Base_dataETTm1_dim7_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-30000.pth \
    --pretrain_args_path pretrain-library/U2METTm1-Base_dataETTm1_dim7_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
    --in_len 336 --out_len $out_len \
    --finetune_layers 1 --neighbor_num 4 --dropout 0.2 --learning_rate 1e-5 --tolerance 3 \
    --gpu 6 --is_training 1
done