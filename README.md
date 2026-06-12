# 3 Input XNOR Gate Neural Network
A simple neural network written from scratch to understand the fundamentals of a neural network. This one predicts the output of a 3-input Exclusive NOR (XNOR) gate.

| Input A | Input B | Input C | XNOR Output (Y) |
| :---:   | :---:   | :---:   | :---:          |
|    0    |    0    |    0    |       **1** |
|    0    |    0    |    1    |       **0** |
|    0    |    1    |    0    |       **0** |
|    0    |    1    |    1    |       **0** |
|    1    |    0    |    0    |       **0** |
|    1    |    0    |    1    |       **0** |
|    1    |    1    |    0    |       **0** |
|    1    |    1    |    1    |       **1** |

## Overview
This is a Neural Network built entirely from scratch only with Python and numpy. It is an MLP that is trained on the 3 input XNOR values and outputs. 
This is my first ever project in Machine Learning and it helped me understand a lot about what goes on under the hood of LLMs and AI Models in general.
XNOR Gates are used in a number of application in both software and hardware, so this is a good start for beginners.

## Network Architecture
* **Input Layers** : 3 (represent the three inputs)
* **Hidden Layers** : 4 (uses the Sigmoid Activation Function)
* **Output Layers** : 1 (for a single output, a 0 or a 1)
* **Weights** : 'w1' and 'w2'

## Mathematical Framework
* **Activation Function** : Sigmoid Function (to squeeze values between 0 and 1)
* **Gradients** : Backpropagation uses the derivative of Sigmoid Function
* **Optimization** : The network minimizes Mean Squared Error (MSE) using Gradient Descent and Backpropagation

## Output
1. Expected and Predicted (Model's answer) Outputs
2. Sum Squared Loss
3. Two txt files with data on the Weights used and one SumSquaredLossList.csv file
4. **Graphs**
   - Expected vs Predicted Outputs
   - Distribution of Predicted Outputs
   - Weights of Layer 1
   - Weights of Layer 2

> The preset training Epochs is 1000. Underneath that is a comment where you can run 10000 epochs and see what happens. Alternately, once you've downloaded it you can run any number epochs.
> For this particular model when I ran it at 50000 epochs it began to overfit and was evident in the weights graphs.

## Prerequistes
This project uses Python 3.
To run it, you must have these modules on your venv or system:
* `numpy`
* `pandas`
* `matplotlib` for pyplot
* `seaborn`

```bash
pip install numpy pandas matplotlib seaborn
```
> NOTE: Mac users, please replace pip with pip3

To clone this repository:
```bash
git clone https://github.com/shadow-edge9/First-Neural-Network
cd First-Neural-Network
python3 main2graphs.py
```

## Usage
This was a tutorial I followed from a book. You can do whatever you want with it :)







