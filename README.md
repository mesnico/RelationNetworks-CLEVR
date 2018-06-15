# Relation Networks CLEVR
A pytorch implementation for _A simple neural network module for relational reasoning_ [https://arxiv.org/abs/1706.01427](https://arxiv.org/abs/1706.01427), working on the CLEVR dataset.

This code tries to reproduce results obtained by DeepMind team, both for the _From Pixels_ and _State Descriptions_ versions they described. Since the paper does not expose all the network details, there could be variations to respect the original results.

The model can also be trained with a slightly modified version of RN, called IR, that enables relational features extraction in order to perform Relational Content Based Image Retrieval (R-CBIR).

## Accuracy

Accuracy values measured on the test set:
  
| Model               |               |
| ------------------- |:-------------:|
| _From Pixels_       | 93.6%         |
| _State Descriptions_ | 97.9%         |

## Get ready
1. Download and extract CLEVR_v1.0 dataset: http://cs.stanford.edu/people/jcjohns/clevr/
2. Clone this repository and move into it:
```
git clone https://github.com/mesnico/RelationNetworks-CLEVR
cd RelationNetworks-CLEVR
```
3. Install requirements: 
```
pip3 install -r requirements.txt
```

## Train

The training code can be run both using Docker or standard python installation with pytorch.
If Docker is used, an image is built with all needed dependencies and it can be easily run inside a Docker container.

### State-descriptions version
Move to the cloned directory and issue the command:
```sh
python3 train.py --clevr-dir path/to/CLEVR_v1.0/ --model 'original-sd' | tee logfile.log
```
We reached an accuracy around 98% over the test set.
Using these parameters, training is performed by using an exponential increase policy for the learning rate (slow start method). Without this policy, our training stopped at around 70% accuracy.
Our training curve measured on the test set:

![accuracy](https://user-images.githubusercontent.com/25117311/39134913-0e02b028-4718-11e8-9bfc-d586962f0b1d.png)

### From-pixels version
Move to the cloned directory and issue the command:
```sh
python3 train.py --clevr-dir path/to/CLEVR_v1.0/ --model 'original-fp' | tee logfile.log
```
We used the same exponential increase policy we employed for the _State Descriptions_ version. We were able to reach around 93% accuracy over the test set:

![accuracy](https://user-images.githubusercontent.com/25117311/40773127-38240290-64c2-11e8-8e58-a989a390d6a9.png)

### Configuration file
We prepared a json-coded configuration file from which model hyperparameters can be tuned. The option ```--config``` specifies a json configuration file, while the option ```--model``` loads a specific hyperparameters configuration defined in the file.
By default, the configuration file is ```config.json``` and the default model is ```original-fp```.
### Training plots
Once training ends, some plots (_invalid answers_, _training loss_, _test loss_, _test accuracy_) can be generated using the ```plot.py``` script:
```
python3 plot.py -i -trl -tsl -a logfile.log
```
These plots are also saved inside ```img/``` folder.

To explore a bunch of other possible arguments useful to customize training, issue the command:
```sh
$ python3 train.py --help
```

## Test
It is possible to run a test session even after training, by loading a specific checkpoint from the trained network collected at a certain epoch. This is possible by specifying the option ```--test```: 
```
python3 train.py --clevr-dir path/to/CLEVR_v1.0/ --model 'original-fp' --resume RN_epoch_xxx.pth --test
```

### Confusion plot
Once test has been performed at least once (note that a test session can be explicitly run but it is also always run automatically after every train epoch), some insights are saved into ```test_results``` and a confusion plot can be generated from them:
```
python3 confusionplot.py test_results/test.pickle
```
![confusion](https://user-images.githubusercontent.com/25117311/40371199-3d78f980-5de2-11e8-8c1f-478e908c19d8.png)

This is useful to discover network weaknesses and possibly solve them.
This plot is also saved inside ```img/``` folder.


## Implementation details
* Questions and answers dictionaries are built from data in training set, so the model will not work with words never seen before.
* All the words in the dataset are treated in a case-insensitive manner, since we don't want the model to learn case biases.
* For network settings, see sections 4 and B from the original paper https://arxiv.org/abs/1706.01427.

## Acknowledgements
Special thanks to https://github.com/aelnouby and https://github.com/rosinality for their great support.
Following, their Relation Network repositories working on CLEVR:
- https://github.com/aelnouby/Relational-Networks
- https://github.com/rosinality/relation-networks-pytorch
