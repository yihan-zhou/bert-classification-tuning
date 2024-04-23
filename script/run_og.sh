# Define variables
batch_size="16"
learning_rate="5e-5"
pre_seq_len="4"
dropout="0.1"
epochs=10


# Run command with all options included
python main.py \
    --model_name_or_path bert-base-cased \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 256 \
    --per_device_train_batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_train_epochs $epochs \
    --pre_seq_len $pre_seq_len \
    --output_dir checkpoints/prefix/ \
    --overwrite_output_dir \
    --hidden_dropout_prob $dropout \
    --seed 11 \
    --load_best_model_at_end True \
    --save_strategy epoch \
    --evaluation_strategy epoch \

  