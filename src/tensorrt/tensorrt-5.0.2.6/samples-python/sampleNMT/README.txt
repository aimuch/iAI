The NMT sample is using data fetched and trained using the NMT tutorial ( https://github.com/tensorflow/nmt ).


** Preparing the data **

The trained weights, directly usable by the sample, can be fetched from here https://developer.download.nvidia.com/compute/machine-learning/tensorrt/models/sampleNMT_weights.tar.gz
'deen/weights' directory should contain all the weight data needed.

We do not distribute the text and vocabulary data. For the De-En model ( https://github.com/tensorflow/nmt#wmt-german-english ), the data needs to be fetched and generated using the following script
https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh . It might take some time, since it prepares 4.5M samples dataset for training as well.
 * Execute wmt16_en_de.sh and it will create 'wmt16_de_en' directory in the current directory
 * 'cd wmt16_de_en'
 * 'cp newstest2015.tok.bpe.32000.de  newstest2015.tok.bpe.32000.en  vocab.bpe.32000.de  vocab.bpe.32000.en <path_to_data_directory>/deen/.'


** Running the sample **

 * List all options supported: sample_nmt --help
 * Run the sample to generate 'translation_output.txt' : sample_nmt --data_dir=<path_to_data_directory>/deen --data_writer=text
 * Get the BLEU score for the first 100 sentences : sample_nmt --data_dir=<path_to_data_directory>/deen --max_inference_samples=100


** Training De-En model using Tensorflow NMT framework and importing the weight data into the sample. **

This section is only relevant if one decides to train the model. 

 * The training data set needs to be fetched and preprocessed as was discussed earlier.
 * Fetch NMT framework : 'git clone https://github.com/tensorflow/nmt.git'
 * Take a look at 'nmt/nmt/standard_hparams/wmt16.json'
   The sample currently only implements unidirectional LSTMs and Luong's attention. So, training should account for this.
   edit relevant JSON config to have {"attention": "luong", "encoder_type": "uni", ...}
   Below is the config we used for training:
    {
      "attention": "luong",
      "attention_architecture": "standard",
      "batch_size": 128,
      "colocate_gradients_with_ops": true,
      "dropout": 0.2,
      "encoder_type": "uni",
      "eos": "</s>",
      "forget_bias": 1.0,
      "infer_batch_size": 32,
      "init_weight": 0.1,
      "learning_rate": 1.0,
      "max_gradient_norm": 5.0,
      "metrics": ["bleu"],
      "num_buckets": 5,
      "num_layers": 2,
      "num_train_steps": 340000,
      "decay_scheme": "luong10",
      "num_units": 1024,
      "optimizer": "sgd",
      "residual": false,
      "share_vocab": false,
      "subword_option": "bpe",
      "sos": "<s>",
      "src_max_len": 50,
      "src_max_len_infer": null,
      "steps_per_external_eval": null,
      "steps_per_stats": 100,
      "tgt_max_len": 50,
      "tgt_max_len_infer": null,
      "time_major": true,
      "unit_type": "lstm",
      "beam_width": 10
    }

 The following line can be used for training, provided the training dataset is /tmp/wmt16_de_en:

    python -m nmt.nmt \
    --src=de --tgt=en \
    --hparams_path=<path_to_json_config>/wmt16.json \
    --out_dir=/tmp/deen_nmt \
    --vocab_prefix=/tmp/wmt16_de_en/vocab.bpe.32000 \
    --train_prefix=/tmp/wmt16_de_en/train.tok.clean.bpe.32000 \
    --dev_prefix=/tmp/wmt16_de_en/newstest2013.tok.bpe.32000 \
    --test_prefix=/tmp/wmt16_de_en/newstest2015.tok.bpe.32000
    
 The following line can be used for the inference in Tensorflow:
    python -m nmt.nmt \
        --src=de --tgt=en \
        --ckpt=/tmp/deen_nmt/translate.ckpt-340000 \
        --hparams_path=<path_to_json_config>/wmt16.json \
        --out_dir=/tmp/deen \
        --vocab_prefix=/tmp/wmt16_de_en/vocab.bpe.32000 \
        --inference_input_file=/tmp/wmt16_de_en/newstest2015.tok.bpe.32000.de \
        --inference_output_file=/tmp/deen/output_infer \
        --inference_ref_file=/tmp/wmt16_de_en/newstest2015.tok.bpe.32000.en
 
 * Importing Tensorflow checkpoint into the sample *
 
 We provide a tool to convert Tensorflow checkpoint from the NMT framework into binary weight data, readable by the sample. It was tested using Tensorflow 1.6. The tool by default imports the NMT framework.
 
     * git clone https://github.com/tensorflow/nmt.git
     * python ./chptToBin.py \
        --src=de --tgt=en \
        --ckpt=/tmp/deen_nmt/translate.ckpt-340000 \
        --hparams_path=<path_to_json_config>/wmt16.json \
        --out_dir=/tmp/deen \
        --vocab_prefix=/tmp/wmt16_de_en/vocab.bpe.32000 \
        --inference_input_file=/tmp/wmt16_de_en/newstest2015.tok.bpe.32000.de \
        --inference_output_file=/tmp/deen/output_infer \
        --inference_ref_file=/tmp/wmt16_de_en/newstest2015.tok.bpe.32000.en