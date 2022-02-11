This project is based on OpenNMT (https://arxiv.org/pdf/1805.11462). Part of this file is from OpenNMT readme file.

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

## Quickstart

[Full Documentation](http://opennmt.net/OpenNMT-py/)


### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

Sequence length can be specified by '-src_seq_length' and '-tgt_seq_length' options.

### Step 2: Train the model

```bash
python train.py -data data/demo -save_model demo-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder.
If you want to train on GPU, you need to set, as an example:
CUDA_VISIBLE_DEVICES=1,3
`-world_size 2 -gpu_ranks 0 1` to use (say) GPU 1 and 3 on this node only.

An example command in our experiment is 'python train.py -data tree_data/parent123/parent123 -save_model tree_logs/parent123/parent123-model -batch_size 1024 -layers 2 --gpu_ranks 0 --learning_rate 0.1 --valid_steps 100 -word_vec_size 200 -rnn_size 200 -encoder_type transformer -decoder_type transformer --position_encoding --train_steps 100000 > tree_logs/parent123/logs1 2>&1'


