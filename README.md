# Seq2Seq

## Usage of Couplet

### Basic Seq2Seq

```
screen -t couplet_seq2seq -S couplet_seq2seq -L
CUDA_VISIBLE_DEVICES=0 sh couplet_seq2seq.sh
```

### Seq2Seq Attention

```
screen -t couplet_seq2seq_attention -S couplet_seq2seq_attention -L
CUDA_VISIBLE_DEVICES=1 sh couplet_seq2seq_attention.sh
```

## Usage of LCSTS

```
screen -t lcsts_word_seq2seq_attention -S lcsts_word_seq2seq_attention -L
CUDA_VISIBLE_DEVICES=1 sh lcsts_word_seq2seq_attention.sh
screen -t lcsts_word_pointer_generator -S lcsts_word_pointer_generator -L
CUDA_VISIBLE_DEVICES=0 sh lcsts_word_pointer_generator.sh
screen -t lcsts_word_seq2seq -S lcsts_word_seq2seq -L
CUDA_VISIBLE_DEVICES=3 sh lcsts_word_seq2seq.sh
screen -t lcsts_char_seq2seq_attention -S lcsts_char_seq2seq_attention -L
CUDA_VISIBLE_DEVICES=0 sh lcsts_char_seq2seq_attention.sh
screen -t lcsts_word_pointer_generator_coverage -S lcsts_word_pointer_generator_coverage -L
CUDA_VISIBLE_DEVICES=3 sh lcsts_word_pointer_generator_coverage.sh
```