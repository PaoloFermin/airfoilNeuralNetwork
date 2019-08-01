# Airfoil Parameter Variation and Prediction

This directory contains files which were used in the development of a MLP Regression Neural Network that predicted the value of Cd and Cl for an airfoil given the velocity and angle of attack. 

## Setting up the environment

Several packages must be installed before using this directory. First and foremost, python's pip package manager must be installed. 

```
sudo apt-get python-pip 
```

Once pip is installed, the syntax to install python packages is 

```
pip install {package}
```

The required packages are:
- numpy
- pandas
- scikit-learn
- matplotlib
- pyfoam
- keras
- tensorflow
- talos


## Running OpenFOAM simulations with PyFoam

To create training data, run

```
python parameter\_variation\_airfoil.py
```

There are two lists within this script which will let you specify the values for U and AoA. The script uses the PyFoam library to modify these values within `0/U` of the airFoil2D "base" OpenFOAM directory, create a copy of that directory, and run simpleFoam from within the copy. The results are scraped from the `postprocessing/` folder of each "copy" directory and stored in `results.csv` within the parent directory. Once the test is done, the copy directories can be deleted for cleanliness. 


## Training the network with Tensorflow, Keras, and Talos

To run the neural network training, run

```
python train_network.py
```

This script first splits the data from `results.csv` into training and validation data. The validation data is stored separately in `test_data.csv` and not used within training. Then the script uses the Talos library to iterate through the training of neural networks with varying hyperparameters. The hyperparameters are specified in the `p` dictionary within the script, and the function `Scan()` performs the iterative training. Once complete, the net with the lowest loss function will be stored in a zip folder `optimized_airfoil_nn_{case number}.zip`. 


## Visualization and Testing

The progress of training can be visualized nicely by running 

```
tensorboard --logdir logs_nn
```

If this is successful, it will display a port which you can connect to through the browser to visualize the training. Typically this is `localhost:6006/`. Each iteration of neural network is stored in a separate numbered directory within the `logs_nn` directory, and thus tensorboard dedicates a separate loss and val_loss graph to each. 

The aforementioned zip folder produced from training contains all relevant files to rebuild the "best" network within a new script, which is exactly what happens when running 

```
python test_network.py
```

This script will access the test data and the most recent network produced by the `train_network.py` script. It will then invoke the network to make predictions on the test data, and compare it with the original results. A graph will be produced for each target variable, showing visually the discrepancy between prediction and actual, as well as a table which will also provide more precise percent error measurements. 


