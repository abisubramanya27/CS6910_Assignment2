# PART B : Using Pre-trained models

**NOTE :** The program is written in a modular manner, with each logically separate unit of code written as functions.  

## Requirements
All the python libraries required to run the program on a CPU are listed in `requirementsB.txt`
They can be installed using 
```shell
pip install -r requirementsB.txt
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
- The code is done in a Google colab notebook and stored in the path `src/Assignment1.ipynb`. It can be opened and run in Google colab or jupyter server.
- The solution to each question is made in the form of a couple of function calls (which are clearly mentioned with question numbers in comments) and commented out in the program so that the user can choose which parts to run and evaluate.
- In order to run the solution for a particular question, uncomment that part and run the cell.
- To get a new model for part B using a pre-trained model (after modifying appropriately and freezing initial layers) use 
  ```python
  get_pretrained_model(model_name, no_neurons_dense, k_value, no_classes)
  ```
  whose arguments are :
  ```python
  model_name -- (string) name of pre-trained model
  no_neurons_dense -- (int) number of neurons in the dense layer
  k_value -- (int) number of layers to freeze in the pre-trained model
  no_classes -- (int) number of output classes in the classification problem (default : 10)
  ```
  and returns :
  ```python
  model -- (Keras Model object) the resulting model
  ```
- Once the model is obtained, we can train it using
```python
CNN_pretrained_train(model, inp_img_shape, train_data_path, config, no_classes, val_data_path, test_data_path, wandb_init)
```
whose arguments are :
  ```python
  model -- (Keras Model object) the pre trained model used for transfer learning
  inp_img_shape -- (tuple) shape of input image
  train_data_path -- (string) the path to training data set
  config -- (dictionary) contains all the hyperparameter and architectural configurations used for the model 
  no_classes -- (int) Number of output classes in the classification problem (default : 10)
  val_data_path -- (string) the path to validation data set (default : None)
  test_data_path -- (string) the path to test data set. If None, would not evaluate the model on test data set (default : None)
  wandb_init -- (bool) True : WANDB run has been initiated outside the function | False : WANDB run not initiated (default : True)
  ```
  and returns :
  ```python
  model -- (Keras Model object) the CNN model which was used for training and/or testing
  id -- (string) the unique run ID from WANDB
  ```
  (**NOTE :** Though val_data_path is set as optional argument, the function won't run without validation data set)
- The <b>config</b> parameter mentioned above is passed to WANDB to log the information related to a model used in a run. It is a python dictionary with the following (necessary) keys :
  ```python
  "learning_rate" --  Hyperparameter for updating the parameters in gradient descent
  "epochs" --  Number of epochs to train the model   
  "optimizer" --  Gradient descent algorithm used for the parameter updation
  "batch_size" --  Batch size used for the optimizer
  "loss_function" --  Loss function used in the optimizer
  "no_neurons_dense" --  Number of neurons in the dense FC layer
  "data_augmented" --  True : Data augmentation is done during training | False : No data augmentation done
  "model_name" --  Name of the pretrained model
  ```
