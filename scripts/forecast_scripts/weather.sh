for out_len in 96 192 336 720
do
    #immediate reaction forecast
    python run_forecast.py --data_format csv --data_name weather --root_path ./datasets/weather/ --data_path weather.csv \
    --data_split 0.7,0.1,0.2 --checkpoints ./forecast_checkpoints/ \
    --pretrained_model_path pretrain-library/U2MWeather-Base_dataweather_dim21_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-60000.pth \
    --pretrain_args_path pretrain-library/U2MWeather-Base_dataweather_dim21_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
    --in_len 336 --out_len $out_len \
    --gpu 0 --is_training 0 --IR_mode

    #finetune
    python run_forecast.py --data_format csv --data_name weather --root_path ./datasets/weather/ --data_path weather.csv \
    --data_split 0.7,0.1,0.2 --checkpoints ./forecast_checkpoints/ \
    --pretrained_model_path pretrain-library/U2MWeather-Base_dataweather_dim21_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-60000.pth \
    --pretrain_args_path pretrain-library/U2MWeather-Base_dataweather_dim21_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json \
    --in_len 336 --out_len $out_len \
    --finetune_layers 1 --neighbor_num 10 --dropout 0.2 --learning_rate 1e-5 --tolerance 3 \
    --gpu 0 --is_training 1
done