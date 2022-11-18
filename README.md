# Charity Funding Predictor

## Skills/Languages Used: Python, Google Colab, sklearn, train_test_split, StandardScaler, pandas, tensorflow, files (from Google Colab)

Using neural networks, and machine learning a model was formed to help select applicants for funding that have the best chance of success in their ventures.

Use this CSV containing more than 34,000 organizations that have received funding from Alphabet Soup in the model. The CSV contains:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

### Step 1: Preprocess the Data

Using pandas and scikit-learn's `StandardScaler()` preprocess the dataset. This will prepare the model for compiling, training, and evaluating the neural network model.

To recreate the model in the Google Colab sheet provided in this repository:

1. Read in the `charity_data.csv` to a Pandas DataFrame

2. Drop the `EIN` and `NAME` columns.

3. Determine the number of unique values for each column.

4. Some columns have more than 10 unique values, cut them down to the size you want to work with. Example `APPLICATION_TYPE` was cut to greater than `528` and `CLASSIFICATION` was cut to ones greater than `1882`.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.

6. Use `pd.get_dummies()` to encode categorical variables.

### Step 2: Compile, Train, and Evaluate the Model

Using TensorFlow, design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Deteremine how many inputs and number of neurons and layers to have in the model.Then compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.

1. Continue using the Google Colab notebook from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export the results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

Using TensorFlow, optimize the model to achieve a target predictive accuracy higher than 75%.

Using any or all of the following methods to optimize the model:

* Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  * Dropping more or fewer columns.
  * Creating more bins for rare occurrences in columns.
  * Increasing or decreasing the number of values for each bin.
* Add more neurons to a hidden layer.
* Add more hidden layers.
* Use different activation functions for the hidden layers.
* Add or reduce the number of epochs to the training regimen.

1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimzation.ipynb`.

2. Import your dependencies and read in the `charity_data.csv` to a Pandas DataFrame.

3. Preprocess the dataset like you did in Step 1, Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Report on the Neural Network Model

The report contains the following:

1. **Overview** of the analysis: Explain the purpose of this analysis.

2. **Results**:

  * Data Preprocessing
    * What variable(s) are the target(s) for your model?
    * What variable(s) are the features for your model?
    * What variable(s) should be removed from the input data because they are neither targets nor features?

* Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take in your attempts to increase model performance?

3. **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

