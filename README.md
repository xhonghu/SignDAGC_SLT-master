## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.
- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.
- For these who failed install ctcdecode (and it always does), you can download [ctcdecode here](https://drive.google.com/file/d/1LjbJz60GzT4qK6WW59SIB1Zi6Sy84wOS/view?usp=sharing), unzip it, and try `cd ctcdecode` and `pip install .`
- Pealse follow [this link](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install pytorch geometric
- You can install other required modules by conducting
  `pip install -r requirements.txt`
  `pip install transformers`

## Data Preparation

1. PHOENIX2014-T datasetDownload the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
3. CSL dataset： Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

Download datasets and extract them, no further data preprocessing needed.

## Pretrained Models

1. *mbart_de* / *mbart_zh* : [pretrained language models](https://drive.google.com/drive/folders/1u7uhrwaBL6sNqscFerJLUHjwt1kuwWw9?usp=drive_link) used to initialize the translation network for German and Chinese, with weights from [mbart-cc-25](https://huggingface.co/facebook/mbart-large-cc25).
2. We provide pretrained models [Phoenix-2014T](https://drive.google.com/drive/folders/1o_fmtmulKlCczz9HaYn0mpvyyCtw-lgs?usp=drive_link) and [CSL-Daily](https://drive.google.com/drive/folders/1IHM49Sp9HRSTvEHe-nf7YeMLm2G1WdS8?usp=drive_link).

Download this directory and place them under *pretrained_models*,The directory structure is as follows..
```
|-- pretrained_models
|   |-- CSL-Daily
|   |   `-- best_model.pt  #Sign language recognition task weight
|   |-- CSL-Daily_g2t
|   |   `-- step_1000.ckpt  #Sign language translation pre-trained weights
|   |-- mBart_de
|   |   |-- config.json
|   |   |-- gloss2ids.pkl
|   |   |-- gloss_embeddings.bin
|   |   |-- map_ids.pkl
|   |   |-- pytorch_model.bin
|   |   |-- sentencepiece.bpe.model
|   |   `-- tokenizer.json
|   |-- mBart_zh
|   |   |-- config.json
|   |   |-- gloss2ids.pkl
|   |   |-- gloss_embeddings.bin
|   |   |-- old2new_vocab.pkl
|   |   |-- pytorch_model.bin
|   |   |-- sentence.bpe.model
|   |   `-- sentencepiece.bpe.model
|   |-- phoenix-2014T
|   |   `-- best_model.pt  #Sign language recognition task weight
|   `-- phoenix-2014T_g2t
|       `-- best.ckpt   #Sign language translation pre-trained weights
```

## Weights

Here we provide the performance of the model (On Test) and its corresponding weights.

| Dataset    | Backbone | Rouge | BLEU1 | BLEU2 | BLEU3 | BLEU4 | Pretrained model                                                                                                          |
| ---------- | -------- | ----- | ----- | ----- | ----- | ----- | ------------------------------------------------------------------------------------------------------------------------- |
| Phoenix14T | Resnet34 | 53.01 | 54.85 | 42.28 | 34.24 | 28.68 | [[Google Drive]](https://drive.google.com/drive/folders/1CfLEpCqvERX7_0AvxzqnquGJEFw-Xg7k?dmr=1&ec=wgc-drive-globalnav-goto) |
| CSL-Daily  | Resnet34 | 52.86 | 55.87 | 42.22 | 32.70 | 25.90 | [[Google Drive]](https://drive.google.com/drive/folders/1PHPewcBlzFrAZmBa8Lh4jBYOYVmEImxH?dmr=1&ec=wgc-drive-globalnav-goto) |

## Evaluate

To evaluate the pretrained model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：

`python main.py --load-weights path_to_weight.pt --phase test`
```
python main.py --load-weights ./phoenix2014-T/best_model.pt --phase test

python main.py --load-weights ./csl-daily/best_model.pt --phase test
```

## Training

To Training the SignDAGC model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：

`python main.py `

Multi-machine training (In fact, the results of the Multi-machine run are not good):

`python -m torch.distributed.launch --nproc_per_node=2 main.py --device 0,1`

## Acknowledgments

Our code is based on [SignGraph](https://github.com/gswycf/SignGraph) and [GreedyViG](https://github.com/SLDGroup/GreedyViG) and [TwoStream](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork).
