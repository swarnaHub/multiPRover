# multiPRover
PyTorch code for our NAACL 2021 paper:

[multiPRover: Generating Multiple Proofs for Improved Interpretability in Rule Reasoning](https://arxiv.org/abs/2106.01354)

[Swarnadeep Saha](https://swarnahub.github.io/), [Prateek Yadav](https://prateek-yadav.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

## Installation
This repository is tested on Python 3.8.3.  
You should install multiPRover on a virtual environment. All dependencies can be installed as follows:
```
pip install -r requirements.txt
```

## Download Dataset
Download the dataset as follows:
```
bash scripts/download_data.sh
```

## Training Iterative-multiPRover
Iterative-multiPRover can be trained by running the following script:
```
bash scripts/train_iterative_mprover.sh
```
This will train on the ```depth-5``` dataset.
The trained model folder will be saved inside ```output``` folder.

## Testing Iterative-multiPRover

The trained Iterative-multiPRover model can be tested by running the following script:
```
bash scripts/test_iterative_mprover.sh
```
This will output the QA accuracy, save the node predictions at ```prediction_nodes_dev.lst``` and the predicted edge logits at ```prediction_edge_logits_dev.lst```.

## Running ILP Inference

Once the node predictions and the edge logits are saved, you can run ILP inference to get edge predictions as follows:
```
bash scripts/run_inference_mprover.sh
```
This will save the edge predictions inside the model folder.

## Evaluation

Once QA, node and edge predictions are saved, you can compute all metrics (QA accuracy, Node F1, Edge F1, Proof F1 and Full accuracy) as follows:
```
bash scripts/get_results_mprover.sh
```

## Zero-shot Evaluation on Birds-Electricity
Run the above testing, inference and evaluation scripts to test the depth-5 trained Iterative-mPRover model on the Birds-Electricity dataset by appropriately changing the ```data-dir``` path to ```data/birds-electricity``` in all the scripts and lines 197 and 198 in ```utils.py``` with ```test.jsonl``` and ```meta-test.jsonl```.


## Training Multilabel-mPRover
Run the following scripts (similar steps as before but changing the paths wherever applicable):
```
bash scripts/train_multilabel_mprover.sh
bash scripts/test_multilabel_mprover.sh
bash scripts/run_inference_mprover.sh
bash scripts/get_results_mprover.sh
```

## Trained Models
We also release our trained multiPRover models on depth-5 dataset [here](https://drive.google.com/file/d/1bIKOmq29teXP87o1KjS3TV1x0caqs6AU/view?usp=sharing). These contain the respective QA, node and edge predictions and you can reproduce the exact validation set results from the paper by running the evaluation script.


## Running Other Ablations
Ablation models from the paper can be run by uncommenting parts of the code (like choosing a particular depth). Please refer to the comments in utils_iterative_mprover.py for details.


### Citation
```
@inproceedings{saha2021multiprover,
  title={multi{PR}over: Generating Multiple Proofs for Improved Interpretability in Rule Reasoning},
  author={Saha, Swarnadeep and Yadav, Prateek and Bansal, Mohit},
  booktitle={NAACL},
  year={2021}
}
```

### Related Citation
multiPRover builds on top of our previous work, [PRover](https://arxiv.org/abs/2010.02830). Feel free to check it out and cite, if you find the paper or the code helpful.
```
@inproceedings{saha2020prover,
  title={{PR}over: Proof Generation for Interpretable Reasoning over Rules},
  author={Saha, Swarnadeep and Ghosh, Sayan and Srivastava, Shashank and Bansal, Mohit},
  booktitle={EMNLP},
  year={2020}
}
```
