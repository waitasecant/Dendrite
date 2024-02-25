## Basic structure of the code

### Helper Functions

1. `getTarget()` : Retrurns target variable and the kind of prediction to be done.

2. `transformStrToModelObjParams()` : Returns a scikit-learn object with standard attribute values and a dictionary containing hyper parameters.

3. `getAlgorithm()` : Returns two dictionaries with scikit-learn objects and hyper parameters for all the relevant models.

4. `featureHandling()` : Returns a data frame after completing imputations.

5. `featureReduction()` : Returns the data matrix after feature reduction and the target variable.

6. ` main()` : Prints best (hyper)parameters, best score and the corresponding model object.

### Sub-helper functions

Note:  These functions are used in `transformStrToModelObjParams()` to pass standard (not-hyper)paramters.

1. `critDT()`, `splitDT()` : Returns the type of criterion, splitter to use for Decision Tree.

2. `kernelSVM()`, `gammaSVM()` : Returns the type of kernel and value of gamma attribute to use for SVM.

3. `lossSGD()`, `penaltySGD()` : Returns the type of loss function and penalty to use for SGD.

4. `weightKNN()` : Returns the weighing scheme to use for kNN.

### Workflow Explained

Read the JSON file and extract following dictionaries: `session_info`, `target`, `feature_handling`, `feature_reduction`, `hyperparameters`, `algorithms`

Execute `main()` is called which implements following steps:

1. Reads the data as a dataframe`df`.

2. Executes `featureHandling(df)` which modifies `df`.

3. Executes `featureReduction(df)` which returns `X`, `y`

4. Executes `getAlgorithm(algorithm, target)` where `algorithms` and `target` are extracted from JSON.

5. For each relevant model depending on `type` in `target` fit `X`, `y` to `GridSearchCV()`

### Note

1. I have turned `is_selected` to `True` for all the models.

2. I have taken a bit liberty in deciding on which algorithms to execute for given type of prediction and the kind of hyper parameters available in `algorithms` dictionary. For instance, in case of SGD, the loss available as (not-hyper)parameter as `use_logistics` and `use_modified_hubber_loss` are available as legitimate attribute in only `SGDClassifier()` and not `SGDRegressor()`. Hence, I have deciding to run this model only if `type` is set as `classification`.

3. Some parameters are not taken into account. For instance, in case of Neural Networks, lot of parameters are set to zero by default, some them are supposed to be positive. Hence, those are not taken into account. 

4. Instructions to some of the possible modifications to the methods is given as comments in the driver code.