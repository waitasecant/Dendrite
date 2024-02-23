# Importing libraries
import json
import pandas as pd
from striprtf.striprtf import rtf_to_text
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, ElasticNet, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# Helper Functions

# Get the target
def getTarget(targetDict):
    target = targetDict['target']
    regtype = targetDict['type']
    return target, regtype

# Helper for DT
def critDT(algoDict, typ):
    if typ == 'clf':
        if algoDict['DecisionTreeClassifier']['use_gini'] == True:
            return 'gini'
        elif algoDict['DecisionTreeClassifier']['use_entropy'] == True:
            return 'entropy'
        else:
            return 'gini'
    elif typ == 'reg':
        if algoDict['DecisionTreeRegressor']['use_gini'] == True:
            return 'gini'
        elif algoDict['DecisionTreeRegressor']['use_entropy'] == True:
            return 'entropy'
        else:
            return 'gini'

# Helper for DT
def splitDT(algoDict, typ):
    if typ == 'clf':
        if algoDict['DecisionTreeClassifier']['use_best'] == True:
            return 'best'
        elif algoDict['DecisionTreeClassifier']['use_random'] == True:
            return 'random'
        else:
            return 'best'
    elif typ == 'reg':
        if algoDict['DecisionTreeRegressor']['use_best'] == True:
            return 'best'
        elif algoDict['DecisionTreeRegressor']['use_random'] == True:
            return 'random'
        else:
            return 'best'
        
# Helper for SVM
def kernelSVM(algoDict):
    if algoDict['SVM']['linear_kernel'] == True:
        return 'linear'
    elif algoDict['SVM']['polynomial_kernel'] == True:
        return 'poly'
    elif algoDict['SVM']['sigmoid_kernel'] == True:
        return 'sigmoid'
    else:
        return 'rbf'
    
# Helper for SVM
def gammaSVM(algoDict):
    if algoDict['SVM']['auto'] == True:
        return 'auto'
    elif algoDict['SVM']['scale'] == True:
        return 'scale'
    else:
        return 'scale'
    
# Helper for SGD
def lossSGD(algoDict):
    if algoDict['SGD']['use_logistics'] == True:
        return 'log_loss'
    elif algoDict['SGD']['use_modified_huber_loss'] == True:
        return 'modified_huber'
    else:
        return 'hinge'
    
# Helper for SGD
def penaltySGD(algoDict):
    if algoDict['SGD']['use_elastic_net_regularization'] == True:
        return 'elasticnet'
    elif algoDict['SGD']['use_l1_regularization'] == 'on':
        return 'l1'
    else:
        return 'l2'
    
# Helper for kNN
def weightKNN(algoDict):
    if algoDict['KNN']['distance_weighting'] == True:
        return 'distance'
    else:
        return 'uniform'
    
# Take in modelStr to return ScikitLearn object
def transformStrToModelObjParams(modelStr, algoDict, regtype):
    if modelStr == 'RandomForestClassifier':
        return RandomForestClassifier(), {
            'n_estimators' : [algoDict[modelStr]['max_trees']],
            'max_depth' : [algoDict[modelStr]['max_depth']],
            'min_samples_split' : [algoDict[modelStr]['min_samples_per_leaf_min_value']]
        }

    elif modelStr == 'RandomForestRegressor':
        return RandomForestRegressor(), {
            'n_estimators' : [algoDict[modelStr]['max_trees']],
            'max_depth' : [algoDict[modelStr]['max_depth']],
            'min_samples_split' : [algoDict[modelStr]['min_samples_per_leaf_min_value']]
        }
    
    elif modelStr == 'GBTClassifier':
        return GradientBoostingClassifier(), {
            'n_estimators' : algoDict[modelStr]['num_of_BoostingStages'],
            'max_depth' : [algoDict[modelStr]['min_depth'],algoDict[modelStr]['max_depth']]
        }
    
    elif modelStr == 'GBTRegressor':
        return GradientBoostingRegressor(), {
            'n_estimators' : algoDict[modelStr]['num_of_BoostingStages'],
            'max_depth' : [algoDict[modelStr]['min_depth'],algoDict[modelStr]['max_depth']]
        }
    
    elif modelStr == 'LinearRegression':
        return LinearRegression(),{}
    
    elif modelStr == 'LogisticRegression':
        return LogisticRegression(
            penalty='elasticnet',
            solver = 'saga'
        ), {
            'max_iter' : [algoDict[modelStr]['min_iter'], algoDict[modelStr]['max_iter']],
            'l1_ratio' : [algoDict[modelStr]['min_elasticnet'], algoDict[modelStr]['max_elasticnet']]
        }
    
    elif modelStr == 'RidgeRegression':
        if type(algoDict[modelStr]['regularization_term']) in [int, float]:
            return Ridge(
                alpha = algoDict[modelStr]['regularization_term']
            ), {
                'max_iter' : [algoDict[modelStr]['min_iter'], algoDict[modelStr]['max_iter']]
            }
        else:
            return Ridge(), {
                'max_iter' : [algoDict[modelStr]['min_iter'], algoDict[modelStr]['max_iter']]
            }
        
    elif modelStr == 'LassoRegression':
        if type(algoDict[modelStr]['regularization_term']) in [int, float]:
            return Lasso(
                alpha = algoDict[modelStr]['regularization_term']
            ), {
                'max_iter' : [algoDict[modelStr]['min_iter'], algoDict[modelStr]['max_iter']]
            }
        else:
            return Lasso(), {
                'max_iter' : [algoDict[modelStr]['min_iter'], algoDict[modelStr]['max_iter']]
            }
        
    elif modelStr == 'ElasticNetRegression':
        if type(algoDict[modelStr]['regularization_term']) in [int, float]:
            return ElasticNet(
                alpha = algoDict[modelStr]['regularization_term']
            ), {
                'max_iter' : [algoDict[modelStr]['min_iter'], algoDict[modelStr]['max_iter']]
            }
        else:
            return ElasticNet(), {
                'max_iter' : [algoDict[modelStr]['min_iter'], algoDict[modelStr]['max_iter']]
            }
        
    elif modelStr == 'xg_boost':
        if regtype == 'regression':
            if algoDict[modelStr]['dart'] == True:
                return XGBRegressor(
                    booster = 'dart',
                    random_state = algoDict[modelStr]['random_state']
                ), {
                    'max_depth' : algoDict[modelStr]['max_depth_of_tree'],
                    'learning_rate' : algoDict[modelStr]['learningRate'],
                    'gamma' : algoDict[modelStr]['gamma'],
                    'min_child_weight' : algoDict[modelStr]['min_child_weight'],
                    'reg_alpha' : algoDict[modelStr]['l1_regularization'],
                    'reg_lambda' : algoDict[modelStr]['l2_regularization']
                }
            else:
                return XGBRegressor(
                    random_state = algoDict[modelStr]['random_state']
                ), {
                    'max_depth' : algoDict[modelStr]['max_depth_of_tree'],
                    'learning_rate' : algoDict[modelStr]['learningRate'],
                    'gamma' : algoDict[modelStr]['gamma'],
                    'min_child_weight' : algoDict[modelStr]['min_child_weight'],
                    'reg_alpha' : algoDict[modelStr]['l1_regularization'],
                    'reg_lambda' : algoDict[modelStr]['l2_regularization']
                }
        elif regtype == 'classification':
            if algoDict[modelStr]['dart'] == True:
                return XGBClassifier(
                    booster = 'dart',
                    random_state = algoDict[modelStr]['random_state']
                ), {
                    'max_depth' : algoDict[modelStr]['max_depth_of_tree'],
                    'learning_rate' : algoDict[modelStr]['learningRate'],
                    'gamma' : algoDict[modelStr]['gamma'],
                    'min_child_weight' : algoDict[modelStr]['min_child_weight'],
                    'reg_alpha' : algoDict[modelStr]['l1_regularization'],
                    'reg_lambda' : algoDict[modelStr]['l2_regularization']
                }
            else:
                return XGBClassifier(
                    random_state = algoDict[modelStr]['random_state']
                ), {
                    'max_depth' : algoDict[modelStr]['max_depth_of_tree'],
                    'learning_rate' : algoDict[modelStr]['learningRate'],
                    'gamma' : algoDict[modelStr]['gamma'],
                    'min_child_weight' : algoDict[modelStr]['min_child_weight'],
                    'reg_alpha' : algoDict[modelStr]['l1_regularization'],
                    'reg_lambda' : algoDict[modelStr]['l2_regularization']
                }
            
    elif modelStr == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(
            criterion = critDT(algoDict, 'clf'),
            splitter = splitDT(algoDict, 'clf')
        ), {
            'max_depth' : [algoDict[modelStr]['min_depth'], algoDict[modelStr]['max_depth']],
            'min_samples_leaf' : algoDict[modelStr]['min_samples_per_leaf']
        }
    
    elif modelStr == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(
            splitter = splitDT(algoDict, 'reg')
        ), {
            'max_depth' : [algoDict[modelStr]['min_depth'], algoDict[modelStr]['max_depth']],
            'min_samples_leaf' : algoDict[modelStr]['min_samples_per_leaf']
        }
    
    elif modelStr == 'SVM':
        if regtype == 'regression':
            return SVR(
                kernel = kernelSVM(algoDict),
                gamma = gammaSVM(algoDict)
            ), {
                'C' : algoDict[modelStr]['c_value'],
                'tol' : [algoDict[modelStr]['tolerance']],
                'max_iter' : [algoDict[modelStr]['max_iterations']]
            }
        elif regtype == 'classification':
            return SVC(
                kernel = kernelSVM(algoDict),
                gamma = gammaSVM(algoDict)
            ), {
                'C' : algoDict[modelStr]['c_value'],
                'tol' : [algoDict[modelStr]['tolerance']],
                'max_iter' : [algoDict[modelStr]['max_iterations']]
            }
        
    elif modelStr == 'SGD':
        return SGDClassifier(
            loss = lossSGD(algoDict),
            penalty = penaltySGD(algoDict)

        ), {
            'alpha' : algoDict[modelStr]['alpha_value'],
            'tol' : [algoDict[modelStr]['tolerance']]
        }
    
    elif modelStr == 'KNN':
        if regtype == 'regression':
            return KNeighborsRegressor(
                weights = weightKNN(algoDict)
            ), {
                'n_neighbors' : algoDict[modelStr]['k_value']
            }
        elif regtype == 'classification':
            return KNeighborsClassifier(
                weights = weightKNN(algoDict)
            ), {
                'n_neighbors' : algoDict[modelStr]['k_value']
            }
        
    elif modelStr == 'extra_random_trees':
        if regtype == 'regression':
            return ExtraTreesRegressor(), {
                'n_estimators' : algoDict[modelStr]['num_of_trees'],
                'max_depth' : algoDict[modelStr]['max_depth'],
                'min_samples_leaf' : algoDict[modelStr]['min_samples_per_leaf']
            }
        elif regtype == 'classification':
            return ExtraTreesClassifier(), {
                'n_estimators' : algoDict[modelStr]['num_of_trees'],
                'max_depth' : algoDict[modelStr]['max_depth'],
                'min_samples_leaf' : algoDict[modelStr]['min_samples_per_leaf']
            }
        
    elif modelStr == 'neural_network':
        if regtype == 'regression':
            return MLPRegressor(
                alpha = algoDict[modelStr]['alpha_value'],
                tol = algoDict[modelStr]['convergence_tolerance'],
                early_stopping = algoDict[modelStr]['early_stopping'],
                shuffle = algoDict[modelStr]['shuffle_data'],
            ), {
                'hidden_layer_sizes' : algoDict[modelStr]['hidden_layer_sizes']
            }
        elif regtype == 'classification':
            return MLPClassifier(
                alpha = algoDict[modelStr]['alpha_value'],
                tol = algoDict[modelStr]['convergence_tolerance'],
                early_stopping = algoDict[modelStr]['early_stopping'],
                shuffle = algoDict[modelStr]['shuffle_data'],
            ), {
                'hidden_layer_sizes' : algoDict[modelStr]['hidden_layer_sizes']
            }
        
# Use getTarget() and transformStrToModelObjParams() to return model object list
def getAlgorithm(algoDict, targetDict):
    _, regtype = getTarget(targetDict)
    if regtype.lower() == 'regression':
        possibleModels = ['RandomForestRegressor', 'GBTRegressor', 'LinearRegression',
                          'RidgeRegression', 'LassoRegression', 'ElasticNetRegression',
                          'xg_boost', 'DecisionTreeRegressor', 'neural_network',
                          'SVM', 'KNN', 'extra_random_trees']
        selectedModels = []
        for models in possibleModels:
            if algoDict[models]['is_selected'] == True:
                selectedModels.append(models)
        modelsObjDict = {}
        modelsParamDict = {}
        for models in selectedModels:
            modelsObjDict[models], modelsParamDict[models] = transformStrToModelObjParams(models, algorithms, regtype)
        return modelsObjDict, modelsParamDict
        

    elif regtype.lower() == 'classification':
        possibleModels = ['RandomForestClassifier', 'GBTClassifier', 'LogisticRegression',
                          'xg_boost', 'DecisionTreeClassifier', 'neural_network',
                          'SVM', 'SGD', 'KNN', 'extra_random_trees']
        selectedModels = []
        for models in possibleModels:
            if algoDict[models]['is_selected'] == True:
                selectedModels.append(models)
        modelsObjDict = {}
        modelsParamDict = {}
        for models in selectedModels:
            modelsObjDict[models], modelsParamDict[models] = transformStrToModelObjParams(models, algorithms, regtype)
        return modelsObjDict, modelsParamDict
    
# Feature Handling
def featureHandling(df):
    for var, feature in feature_handling.items():
        if feature['is_selected'] == False:
            df.drop(var, axis=1, inplace=True)
        if feature['feature_variable_type'] == 'numerical':
            if feature['feature_details']['missing_values'] == 'Impute':
                if feature['feature_details']['impute_with'] == 'Average of values':
                    df[var].fillna(df[var].mean(), inplace=True)
                elif feature['feature_details']['impute_with'] == 'custom':
                    df[var].fillna(feature['feature_details']['impute_value'], inplace=True)
        elif feature['feature_variable_type'] == 'text':
            df[var] = LabelEncoder().fit_transform(df[var])
    return df

# Feature Reduction
def featureReduction(df):
    targetVar, regtype = getTarget(target)

    X = df.drop(targetVar, axis=1).to_numpy()
    y = df[targetVar].to_numpy()

    if feature_reduction['feature_reduction_method'] == 'Tree-based':
        if regtype == 'regression':
            selector = SelectFromModel(
                RandomForestRegressor(
                n_estimators=int(feature_reduction['num_of_trees']),
                max_depth=int(feature_reduction['depth_of_trees'])
                )).fit(X, y)

        elif regtype == 'classification':
            selector = SelectFromModel(RandomForestClassifier(
                n_estimators=int(feature_reduction['num_of_trees']),
                max_depth=int(feature_reduction['depth_of_trees'])
                )).fit(X, y)
            
        featImp = selector.estimator_.feature_importances_
        featImp = [(j,i) for i,j in enumerate(featImp)]
        featImp.sort(reverse=True)
        featImpIndex = [k for _, k in featImp[:eval(feature_reduction['num_of_features_to_keep'])]]
        
        tempdf = df.iloc[:,featImpIndex]
        return tempdf.to_numpy(), y

    elif feature_reduction['feature_reduction_method'] == 'Principal Component Analysis':
        pca = PCA(n_components=int(feature_reduction['num_of_features_to_keep'])).fit(X)
        X = pca.transform(X)
        return X, y

    elif feature_reduction['feature_reduction_method'] == 'Correlation with target':
        corr = df.corr()[targetVar].drop(targetVar)
        corr = [(j,i) for i,j in zip(corr.index, corr.values)]
        corr.sort(reverse=True)
        corrIndex = [k for _, k in corr[:eval(feature_reduction['num_of_features_to_keep'])]]
        
        tempdf = df.loc[:,corrIndex]
        return tempdf.to_numpy(), y

    elif feature_reduction['feature_reduction_method'] == 'No Reduction':
        return X, y
    
# Main driver function
def main():
    dataset = session_info['dataset']
    df = pd.read_csv(dataset)
    df = featureHandling(df)
    X, y = featureReduction(df)
    modelsObjScikitDict, paramDict = getAlgorithm(algorithms, target)
    for model in paramDict.keys():
        print(model)
        finalModel = GridSearchCV(modelsObjScikitDict[model], paramDict[model], cv=5, n_jobs=-1)
        finalModel.fit(X, y)
        print(f"Best parameters: {finalModel.best_params_}")
        print(f"Best score: {finalModel.best_score_}")
        print(finalModel.best_estimator_)
        print('\n')

if __name__ == "__main__":
    # Reading JSON
    with open('algoparams_from_ui.json.rtf', 'r') as f: 
        rtfText = f.read() 
    plainText = rtf_to_text(rtfText)
    mainDict = json.loads(plainText)

    # Extracting the Dicts in mainDict
    session_info = mainDict['design_state_data']['session_info']
    target = mainDict['design_state_data']['target']
    feature_handling = mainDict['design_state_data']['feature_handling']
    feature_reduction = mainDict['design_state_data']['feature_reduction']
    hyperparameters = mainDict['design_state_data']['hyperparameters']
    algorithms = mainDict['design_state_data']['algorithms']

    # Modifying algorithms to test
    algorithms['RandomForestRegressor']['is_selected'] = True
    algorithms['RandomForestClassifier']['is_selected'] = True
    algorithms['GBTClassifier']['is_selected'] = True
    algorithms['GBTRegressor']['is_selected'] = True
    algorithms['LinearRegression']['is_selected'] = True
    algorithms['LogisticRegression']['is_selected'] = True
    algorithms['RidgeRegression']['is_selected'] = True
    algorithms['LassoRegression']['is_selected'] = True
    algorithms['ElasticNetRegression']['is_selected'] = True
    algorithms['xg_boost']['is_selected'] = True
    algorithms['DecisionTreeRegressor']['is_selected'] = True
    algorithms['DecisionTreeClassifier']['is_selected'] = True
    algorithms['SVM']['is_selected'] = True
    algorithms['SGD']['is_selected'] = True
    algorithms['KNN']['is_selected'] = True
    algorithms['extra_random_trees']['is_selected'] = True
    algorithms['neural_network']['is_selected'] = True
    
    main()