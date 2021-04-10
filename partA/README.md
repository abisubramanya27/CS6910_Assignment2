# PART A : Training CNN from scratch

**NOTE :** The program is written in a modular manner, with each logically separate unit of code written as functions.  

## Requirements
All the python libraries required to run the program on a CPU are listed in `requirementsA.txt` ([link](requirementsA.txt)).
They can be installed using 
```shell
pip install -r requirementsA.txt
```
**(Use Python 3.7 or lower versions of Python 3)**

The training happens much faster on a GPU and the notebook can be run on Google colab without any additional installations than the ones done in the notebook itself.

The very first time you run the notebook, the dataset needs to be downloaded. The inaturalist_12K dataset has already been split into training, validation and testing sets (using `split-folders` python library) and stored in gdrive. It can be downloaded by running the below in the notebook cell 
```python
!gdown --id 11SGStqp8Vug2GDzSpJDwQYHThLIjZFQn
!unzip -q inaturalist_12K.zip
!ls
```
(the above lines are already present in the first cell of the notebook, they need to be uncommented to run)


## Steps to run the program
- The code is done in a Google colab notebook and stored in the path `src/Assignment2_PartA.ipynb` ([link](src/Assignment2_PartA.ipynb)). It can be opened and run in Google colab or jupyter server.
- The solution to each question is made in the form of a couple of function calls (which are clearly mentioned with question numbers in comments) and commented out in the program so that the user can choose which parts to run and evaluate.
- In order to run the solution for a particular question, uncomment that part and run the cell.
- To get a new model for part A use 
  ```python
  build_model_partA(inp_img_shape, K_list, F_list, no_neurons_dense, no_classes, pooling_list, activation_fn_list, P_list, S_list, reg_list, lambda_, BN_yes, dropout_p)
  ```
  whose arguments are :
  ```python
  inp_img_shape -- (int) shape of input image
  K_list -- (list) List of number of filters in each non FC layer
  F_list -- (list of int) List of size of filters (assumed same dimension in width and height) in each non FC layer  
  no_neurons_dense -- (int) Number of neurons in the dense FC layer
  no_classes -- (int) Number of output classes in the classification problem
  pooling_list -- (list of string) List of pooling layer option for each conv+pooling block ('max' : MaxPooling2D, 'avg': AveragePooling2D)
  activation_fn_list -- (list of string) List of activation function in each convolution layer and the one hidden FC layer (name of activation function as required by Keras Activation)
  P_list -- (list of string) List of padding options in each non FC layer 
            ('valid' : no padding, 'same' : padding to make input and output same dimensions)
  S_list -- (list of int) List of strides (assumed equal in width and height) in each non FC layer
  reg_list -- (list of string) List of regularization options for the convolution, one hidden FC and output layers ('none' : no regularization, 'L2' , 'L1')
  lambda_ -- (float) weight decay hyperparameter for regularisation
  BN_yes -- True : (bool) Batch normalisation (BN) should be used, False : BN should not be used
  dropout_p -- (float between 0 and 1) Probability of dropping out a neuron
               [The dropout is added for the single dense hidden layer alone after referring to many CNN architecture papers]
  ```
  and returns :
  ```python
  model -- (Keras Model object) the resulting model
  ```
- To train a CNN and optionally evalute it on a test set use
```python
CNN_train(inp_img_shape, train_data_path, K_list, F_list, config, no_classes, pooling_list, activation_fn_list, P_list, S_list, reg_list, val_data_path, test_data_path, wandb_init, load_run)
```
whose arguments are :
  ```python
  inp_img_shape -- (tuple) shape of input image
  train_data_path -- (string) the path to training data
  K_list -- (list of int) List of number of filters in each non FC layer
  F_list -- (list of int) List of size of filters (assumed same dimension in width and height) in each non FC layer 
  config -- (dictionary) contains all the hyperparameter and architectural configurations used for the model 
            [refer to the config_1 in next cell to see what all it contains]
  no_classes -- (int) Number of output classes in the classification problem
  pooling_list -- (list of string) List of pooling layer option for each conv+pooling block ('max' : MaxPooling2D, 'avg': AveragePooling2D)
  activation_fn_list -- (list of string) List of activation function in each convolution layer and the onne hidden FC layer (name of activation function as required by Keras Activation)
  P_list -- (list of string) List of padding options in each non FC layer 
            ('valid' : no padding, 'same' : padding to make input and output same dimensions)
  S_list -- (list of int) List of strides (assumed equal in width and height) in each non FC layer
  reg_list -- (list of string) List of regularization options for the convolution, one hidden FC and output layers ('none' : no regularization, 'L2' , 'L1')
  val_data_path -- (string) the path to validation data (default : None)
  test_data_path -- (string) the path to test data. If None, would not evaluate the model on test data set (default : None)
  wandb_init -- (bool) True : WANDB run has been initiated outside the function | False : WANDB run not initiated
  load_run -- (string) WANDB run ID to restore and use a previously trained model (None to create a new model)
  ```
  and returns :
  ```python
  model -- (Keras Model object) the CNN model which was used for training and/or testing
  id -- (string) the unique run ID from WANDB
  ```
  (**NOTE :** Though val_data_path is set as optional argument, it is required for training (i.e) if <b>load_run</b> is None)
- The <b>config</b> parameter mentioned above is passed to WANDB to log the information related to a model used in a run. It is a python dictionary with atleast the following necessary keys (in the notebook the lists we passed to <b>CNN_train</b> are also included in the config) :
  ```python
  "learning_rate" --  Hyperparameter for updating the parameters in gradient descent
  "epochs" --  Number of epochs to train the model   
  "optimizer" --  Gradient descent algorithm used for the parameter updation
  "batch_size" --  Batch size used for the optimizer
  "loss_function" --  Loss function used in the optimizer
  "architecture" --  Type of neural network used
  "dataset" --  Name of dataset
  "no_filters" --  Number of filters for the first convolution layer
  "filter_organization" -- The factor by which the number of filters change in the subseqeuent convolution layers
  "no_neurons_dense" --  Number of neurons in the dense FC layer
  "data_augmented" --  True : Data augmentation is done during training, False : No data augmentation done
  "dropout" --  Probability of dropping out a neuron in dropout technique
  "batch_normalization" --  True : Batch normalisation (BN) should be used, False : BN should not be used
  "weight_decay" --  weight decay hyperparameter for regularization
  "input_image_shape" --  shape of input image to the model
  ```
