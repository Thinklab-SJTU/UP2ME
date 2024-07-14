root_path_name=datasets/

export CUDA_VISIBLE_DEVICES=0,1,2,3

data_split='0.7,0.1,0.2'
random_seed=2023
gpu=0
test_random_seed=4069
model="U2M_finetune"
script_name="run_imputation.py"

data_split='34560,11520,11520'

for model in $model; do
  for dataset in "ETTm1.csv"; do
    for in_len in 600; do
      for min_mask_ratio in 0 0.125 0.25 0.375; do
        echo Model_$model'_dataset_'$dataset'_in_len_'$in_len'_mask_ratio_'$min_mask_ratio >> \
          logs/exp/Imputation/Model_$model'_dataset_'$dataset'_in_len_'$in_len'_mask_ratio_'$min_mask_ratio.txt
        python -u $script_name \
          --data_format 'csv' \
          --random_seed $random_seed \
          --test_random_seed $test_random_seed \
          --root_path $root_path_name \
          --data_name $dataset \
          --is_training \
          --in_len $in_len \
          --slide_step $in_len \
          --itr 1 \
          --train_epochs 20 \
          --learning_rate 1e-3 \
          --batch_size 512 \
          --min_mask_ratio $min_mask_ratio \
          --max_mask_ratio $(echo "$min_mask_ratio + 0.125" | bc) \
          --gpu $gpu \
          --data_split $data_split >logs/exp/Imputation/Model_$model'_dataset_'$dataset'_in_len_'$in_len'_mask_ratio_'$min_mask_ratio.txt
      done
    done
  done
done