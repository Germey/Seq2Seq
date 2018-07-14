#!/usr/bin/env bash
cd ../..
python3 inference.py\
    --beam_width 1\
    --inference_batch_size 256\
    --model_path checkpoints/lcsts_word_seq2seq/lcsts.ckpt-386000\
    --inference_input dataset/lcsts/word/sources.test.txt\
    --inference_output dataset/lcsts/word/summaries.inference.txt\