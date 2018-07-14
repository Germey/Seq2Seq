#!/usr/bin/env bash
cd ..
python3 train.py\
    --model_class pointer_generator_coverage\
    --batch_size 256\
    --hidden_units 400\
    --embedding_size 300\
    --attention_units 250\
    --encoder_depth 3\
    --decoder_depth 3\
    --encoder_max_time_steps 80\
    --decoder_max_time_steps 20\
    --display_freq 5\
    --save_freq 2000\
    --valid_freq 400\
    --model_dir checkpoints/lcsts_word_pointer_generator_coverage\
    --model_name lcsts.ckpt\
    --source_vocabulary dataset/lcsts/word/vocabs.json\
    --target_vocabulary dataset/lcsts/word/vocabs.json\
    --source_train_data dataset/lcsts/word/sources.train.txt\
    --target_train_data dataset/lcsts/word/summaries.train.txt\
    --source_valid_data dataset/lcsts/word/sources.eval.txt\
    --target_valid_data dataset/lcsts/word/summaries.eval.txt\
    --encoder_vocab_size 30000\
    --decoder_vocab_size 30000\
    --cell_type gru\
    --max_epochs 100000\
    --extend_vocabs True