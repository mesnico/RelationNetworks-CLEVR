# RelationalNetworksCLEVR
A pytorch implementation for "A simple neural network module for relational reasoning", working on the CLEVR dataset

This code tries to reproduce results obtained by DeepMind team. Since the paper does not expose all the network details, there could be variations to respect the original results.
It is possible to stop training at any time: after every epoch a checkpoint is saved inside the model/ folder automatically created during training.

## Run
The code can be run both using Docker or standard python installation with pytorch.
If Docker is used, an image is built with all needed dependencies and it can be easily run inside a Docker container.
### Without Docker
Simply move to the cloned directory and issue the command

```sh
$ python3 main.py --clevr-dir /path/to/clevr_dataset/CLEVRv1.0
```
To explore a bunch of other possible arguments useful to customize training, issue command
```sh
$ python3 main.py --help
```

### With Docker
To build and execute the docker image, move to the cloned directory and issue the following commands

```sh
$ sudo docker_build.sh
$ sudo docker_run.sh /path/to/clevr_dataset/CLEVR_v1.0
```

By default, the training is performed with mini-batches of size 64 and for a maximum of 1400000 epochs.
If multiple GPUs are used or some specific parameters for the main.py script are needed, it is required to slightly modify the docker_run.sh script.

## Implementation details
* Questions and answers dictionaries are built from data in training set, so the model will not work with words never seen before. This is a problem if the training-set is changed: the model should be retrained.
	* Possible solution: use an English dictionary instead
* All the words in the dataset are treated in a case-insensitive manner, since we don't want the model to learn case biases.
* For network settings, see sections 6B and 6C from the original paper https://arxiv.org/abs/1706.01427. Question embeddings in output from last LSTM layer are assumed to be 32 dimensional features.
