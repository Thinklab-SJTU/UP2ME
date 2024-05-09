python run_pretrain.py --data_format npy --data_name NIPS_Water --root_path ./datasets/NIPS_Water/ --valid_prop 0.2 --checkpoints ./pretrain-library/ \
--data_dim 9 --patch_size 10 --min_patch_num 5 --max_patch_num 100 --mask_ratio 0.5 \
--train_steps 500000 --valid_freq 10000 --valid_batches 1000 --tolerance 10 --gpu 0 --label NIPS_Water-Base

python run_pretrain.py --data_format csv --data_name ETTm1 --root_path ./datasets/ETT/ --data_path ETTm1.csv --data_split 34560,11520,11520 --checkpoints ./pretrain-library/ \
--data_dim 7 --patch_size 12 --min_patch_num 20 --max_patch_num 200 --mask_ratio 0.5 \
--batch_size 256 --train_steps 500000 --valid_freq 5000 --tolerance 10 --gpu 0 --label ETTm1-Base