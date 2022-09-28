# AI-Security-Course

This Repo is mainly for the design and implementation of experiments in AI+Security course in ZJUSE.

<img src="img/myplot.png" alt="failed to load img">

The confusion matrix of CIFAR10 below.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing
purposes.

### Prerequisites

You maybe need Anaconda to create a virtual environment for running python.

```shell
conda create -n name=3.6
source activate name
```

### Installing

Then just install all python package required by our project by one cmd line:

```shell
pip install -r requirements.txt
```

## Running the tests and evaluation

Explain how to run the automated tests for this system

### Generating predicted result

Given a well-trained classify model, we first take the test set of CIFAR10 as the input of the model and obtain
responsive predict labels via the following shell:

```shell
# feel free to change
python src/test.py --path 1234-5678/20.pth --device cuda
```

Make sure you have sufficient right on .sh before run it:

```shell
chmod 777 run_test.sh
./run_test.sh
```

After that you will get a .txt which contains all test result(10,000 digit):

```text
3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6 7 0 4 9 5 2 4 0 9 6 6 5 4 5 9 2 4 1 9 5 4 6 5 6 0 9 3 9 7 6 9 8 0 3 8 8 7
```

0~9 correspond to ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') respectively.

### Evaluate result

Alter run_evaluate.sh as you wish:

```shell
# in run_evaluate.sh
python src/evaluate.py --gt gt.txt --pre prediction.txt
```

Then run the following cmd to obtain evaluation results:

```shell
./run_evaluate.sh
```

## Build your own model

### Change your model

Modify src/model.py

### Train your model

```shell
./run_train.sh
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [PyTorch Tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
* Jie Song VIPA
