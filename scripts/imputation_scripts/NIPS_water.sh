root_path_name=datasets/

export CUDA_VISIBLE_DEVICES=0,1,2

data_split='0.7,0.1,0.2'
random_seed=2023
gpu=2
test_random_seed=4069
model="U2M_finetune"
script_name="run_imputation.py"


for model in $model; do
  for dataset in 'NIPS_Water'; do
    for in_len in 600; do
      for min_mask_ratio in 0 0.125 0.25 0.375; do
        echo Model_$model'_dataset_'$dataset'_in_len_'$in_len'_mask_ratio_'$min_mask_ratio >> \
          logs/exp/Imputation/Model_$model'_dataset_'$dataset'_in_len_'$in_len'_mask_ratio_'$min_mask_ratio.txt
        python -u $script_name \
          --data_format 'npy' \
          --random_seed $random_seed \
          --test_random_seed $test_random_seed \
          --root_path $root_path_name \
          --data_name $dataset \
          --in_len $in_len \
          --slide_step $in_len \
          --itr 1 \
          --is_training \
          --train_epochs 20 \
          --learning_rate 0.0001 \
          --batch_size 128 \
          --gpu $gpu \
          --min_mask_ratio $min_mask_ratio \
          --max_mask_ratio $(echo "$min_mask_ratio + 0.125" | bc) \
          --data_split $data_split >logs/exp/Imputation/Model_$model'_dataset_'$dataset'_in_len_'$in_len'_mask_ratio_'$min_mask_ratio.txt
      done
    done
  done
done
