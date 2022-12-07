




import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve

import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import MinMaxScaler

from ML_tools.classification.hbagging.hbagging import HBagging
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
#from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ML_tools.classification.fit_classification_algorithm import fit_classification
from sklearn.metrics import precision_score, accuracy_score





# def plot_roc_auc_curve():
    
    
    # tprs = []
    # aucs = []
    # mean_fpr = np.linspace(0, 1, 100)


    # fig, ax = plt.subplots(1, figsize=(12,12))
    
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #     label='Random classifier', alpha=.8)

    # ax.set_xlim([-0.05, 1.05])
    # ax.set_ylim([-0.05, 1.05])

       
    # ax.set_ylabel("True positive rate", fontsize=30)
    # ax.set_xlabel("False positive rate", fontsize=30)


    # fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=final_predictions)
    # interp_tpr = np.interp(mean_fpr, fpr, tpr)
    # interp_tpr[0] = 0.0
    # tprs.append(interp_tpr)


    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    
    # std_auc = np.std(aucs)
    # ax.plot(mean_fpr, mean_tpr, color='b',
    #         label=r'Mean ROC curve (AUC = %.2f $\pm$ %.2f)' % (np.mean(auc_test), np.std(auc_test)),
    #         lw=2, alpha=.8)
    
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightgrey', alpha=.2,
    #                 label=r'$\pm$ 1 std. dev.')
    

    # ax.set_aspect("equal")

    # ax.legend(loc="lower right", fontsize=26) 
    
    
    # # Final model
    # ax.tick_params(axis="both",labelsize=20)
    # fig.tight_layout()            
    # fig.savefig(Path(fig_path, "HBagging_roc_curve_"+title+".png"))



def run_ikky_classification_LOO(X, y, algo, leave_one_out_values, variables_names, dic_variables, 
                            params_grid, name_classes, train_prop=0.8, \
                                 n_cross_val_splits=5, \
                                 stratify_=True, plot=True,
                                 nb_classifiers=10, tolerance=0.1, fig_path=None,
                                 reversed_variables_names=None,
                                 rename_dic={},
                                 title="", threshold_importance=0.5):


    variables_names = np.array(variables_names)
   
    auc_test = []
    
    selected_models = {variable_type:[] for variable_type in dic_variables}
    auc_variable_types = {variable_type:[] for variable_type in dic_variables}

    strat = y if stratify_ else None

    index_lists = {}
    
    for variable_type, list_ in dic_variables.items():
        index_list_ = []
        for var in list_:
            ind = list(variables_names).index(var)
            index_list_.append(ind)
        index_lists[variable_type] = np.array(index_list_)


    unique_leave_one_out_values = np.unique(leave_one_out_values)
    

    dic_importances = {var:[] for var in variables_names}
               
    for pilot_to_leave_out in unique_leave_one_out_values:

        predictions_subtest = []
        auc_subtest = []
        
        test_pilot = pilot_to_leave_out
        train_pilots = [k for k in unique_leave_one_out_values if k!=pilot_to_leave_out]
        
        index_train = np.isin(leave_one_out_values, train_pilots)
        index_test = leave_one_out_values == test_pilot
        
        print("train on pilots", train_pilots, "test on pilot", pilot_to_leave_out)
        
        X_train = X[index_train]
        X_test = X[index_test]
        y_train = y[index_train]
        y_test = y[index_test]
        
        if algo=="logistic":         
            # define min max scaler
            scaler = MinMaxScaler()
            # transform data
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
    

        
        for variable_type, list_ in dic_variables.items():
            
            index_list_ = index_lists[variable_type]

            sub_X_train = X_train[:,index_list_]
            sub_X_test = X_test[:,index_list_]
 
            auc_grid = []
            
            fitted_model, importances = fit_classification(sub_X_train, y_train, algo, params_grid, n_cross_val_splits=n_cross_val_splits)
            
            selected_models[variable_type].append(fitted_model)

            prediction_subtest = fitted_model.predict_proba(sub_X_test)[:,1]
            predictions_subtest.append(prediction_subtest)

            #Modifications par Ioannis et Dimitri
            #print(len(y_test))
            auc_score = roc_auc_score(y_true=y_test, y_score=prediction_subtest)
            #auc_score = accuracy_score(np.int32(y_test), np.int32(prediction_subtest)) 
            #Fin modifcations
            auc_subtest.append(auc_score)
                  
            auc_variable_types[variable_type].append(auc_score)

            print(variable_type, "AUC", auc_score)


            for i, var in enumerate(variables_names[index_list_]):
                dic_importances[var].append(importances[i])
                     
#    
#            if plot:
#                model.draw(sub_X_train, y_train, title="training_"+str(k)+variable_type, features_names=variables_names_clean[index_list_], rename_dic=rename_dic, fig_path=fig_path, reversed_variables_names_clean=reversed_variables_names)
#                model.draw(sub_X_test, y_test, title="testing_"+str(k)+variable_type, features_names=variables_names_clean[index_list_], rename_dic=rename_dic, fig_path=fig_path, reversed_variables_names_clean=reversed_variables_names)

        if algo=="hbagging":
            rename_dic = {name:name.replace("feature_tc_fixed_windows_","").replace("_"," ") for name in variables_names}
            rename_dic = {k:v[0].upper()+v[1:] for k,v in rename_dic.items()}
    
            fitted_model.draw(sub_X_test, y_test, title="Hbagging_test_pilot_"+pilot_to_leave_out+"_"+title+"_"+variable_type, features_names=variables_names[index_list_], rename_dic=rename_dic, fig_path=fig_path, reversed_variables_names=reversed_variables_names, \
                             name_classes=name_classes)



        predictions_subtest = np.array(predictions_subtest).T

        final_predictions = np.mean(predictions_subtest, axis=1)

## #ALLO C'EST ICI QU'IL FAUT DECOMMENTER ###
        auc_score = roc_auc_score(y_true=y_test, y_score=final_predictions)
        auc_test.append(auc_score)
  
    
    """ print auc """

    for variable_type, names_ in dic_variables.items():
                
        aucs = auc_variable_types[variable_type]
        print("\n", variable_type, len(index_lists[variable_type]), "%.3f"%np.mean(aucs), "écart-type", "%.3f"%np.std(aucs))
    
    print("\nAUC moyenne sur les ensembles de test :", "%.3f"%np.mean(auc_test), "écart-type", "%.3f"%np.std(auc_test))


    """ print features importance """



    for variable_type, list_ in dic_variables.items():

        if algo=="logistic" or algo=="forest":
            importances = [np.mean(dic_importances[var]) for var in list_]
            
        elif algo=="hbagging":
            importances = [np.mean(dic_importances[var]) for var in list_]
      
        print("--")
        print(variable_type)
        print("--")
        
        indices = np.argsort(np.abs(importances))[::-1]
        sorted_variables = np.array(list_)[indices]
            
        print("Poids des variables dans le modèle :\n", variable_type)

        for i, var in enumerate(sorted_variables):
            variable_importance = dic_importances[var]
            
            if algo=="hbagging" and np.sum(variable_importance)==0:
                continue
            if algo!="hbagging" and np.abs(np.mean(variable_importance)) < threshold_importance*np.max(np.abs(importances)):
                continue

            if algo=="hbagging":
                print(var, np.sum(variable_importance))
            
            else:
                print(var, np.mean(variable_importance))



    """ Gobal model for interpretation only, only for hbagging """
    
    if algo=="hbagging":
            
        for variable_type, list_ in dic_variables.items():
    
            index_list_ = []
            for var in list_:
                ind = list(variables_names).index(var)
                index_list_.append(ind)
            
            index_list_ = np.array(index_list_)
    
            sub_X = X[:,index_list_]

            total_model, variables_used = fit_classification(sub_X, y, algo, params_grid, n_cross_val_splits)

        
        rename_dic = {name:name.replace("feature_tc_fixed_windows_","").replace("_"," ") for name in variables_names}
        rename_dic = {k:v[0].upper()+v[1:] for k,v in rename_dic.items()}
    
        total_model.draw(sub_X, y, title="Hbagging_"+title+"_"+variable_type, features_names=variables_names[index_list_], rename_dic=rename_dic, fig_path=fig_path, reversed_variables_names=reversed_variables_names, \
                             name_classes=name_classes)







# def hbagging_classification_ikky_LOO(X, y, leave_one_out_values, variables_names, dic_variables, 
#                             params_grid, name_classes, train_prop=0.8, \
#                                  num_iterations=10, cross_validations_splits=5, \
#                                  stratify_=True, plot=True,
#                                  nb_classifiers=10, tolerance=0.1, optimize_hyperparameters=False,
#                                  fig_path=None,
#                                  reversed_variables_names=None,
#                                  rename_dic={},
#                                  title=""):


    
#     variables_names = np.array(variables_names)

# #    
# #    
# #    variables_names_clean = np.array([col.replace("walk_features_","").replace("static_features_GENERATIVE_MODEL_","").replace("excel_scores_","").replace("_"," ") for col in variables_names])
# #    reversed_variables_names_clean = np.array([col.replace("walk_features_","").replace("static_features_GENERATIVE_MODEL_","").replace("excel_scores_","").replace("_", " ") for col in reversed_variables_names])
# #        
#     auc_test = []
    
#     selected_models = {variable_type:[] for variable_type in dic_variables}
#     auc_variable_types = {variable_type:[] for variable_type in dic_variables}


#     strat = y if stratify_ else None

#     dic_importances = {var:[] for var in variables_names}


    
    
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)


#     fig, ax = plt.subplots(1, figsize=(12,12))
    
#     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#         label='Random classifier', alpha=.8)

#     ax.set_xlim([-0.05, 1.05])
#     ax.set_ylim([-0.05, 1.05])

        

#     unique_leave_one_out_values = np.unique(leave_one_out_values)
           
#     for pilot_to_leave_out in unique_leave_one_out_values:

#         predictions_subtest = []
#         auc_subtest = []

        
#         test_pilot = pilot_to_leave_out
#         train_pilots = [k for k in unique_leave_one_out_values if k!=pilot_to_leave_out]
        
#         index_train = np.isin(leave_one_out_values, train_pilots)
#         index_test = leave_one_out_values == test_pilot
        
#         print("train on pilots", train_pilots, "test on pilot", pilot_to_leave_out)
        
#         X_train = X[index_train]
#         X_test = X[index_test]
#         y_train = y[index_train]
#         y_test = y[index_test]
        
#         for variable_type, list_ in dic_variables.items():
            
#             index_list_ = []
#             for var in list_:
#                 ind = list(variables_names).index(var)
#                 index_list_.append(ind)

#             index_list_ = np.array(index_list_)

#             sub_X_train = X_train[:,index_list_]
#             sub_X_test = X_test[:,index_list_]

            
#             auc_grid = []
            
#             for nb_classifiers in params_grid["nb_classifiers"]:

#                 for tolerance in params_grid["tolerance"]:
                    
#                     if nb_classifiers > sub_X_train.shape[1]:
#                         auc_grid.append((0, nb_classifiers, tolerance))
#                         continue                                              

#                     model_grid = HBagging()
                    



#                     cross_val_scores = []

#                     for v in range(cross_validations_splits):
                        
#                         X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(sub_X_train, y_train, \
#                                                                             test_size=1-train_prop, \
#                                                                             stratify=y_train,
#                                                                             shuffle=True)
                        

# #                        for train_index, val_index in skf.split(sub_X_train, y_train):
# #                            X_train_, X_val = X[train_index], X[val_index]
# #                            y_train_, y_val = y[train_index], y[val_index]
# #                            

#                         model_grid.fit(X_train_cv, y_train_cv, nb_classifiers=nb_classifiers, tolerance=tolerance)
            
#                         auc_score = roc_auc_score(y_true=y_val_cv, y_score=model_grid.predict_proba(X_val_cv)[:,1])
#                         cross_val_scores.append((auc_score, nb_classifiers, tolerance))  
#                     sorted_auc_cross_val = sorted(cross_val_scores, key=lambda tup:tup[0])[::-1]

#                     best_nb_classifiers_val = sorted_auc_cross_val[0][1]
#                     best_tolerance_val = sorted_auc_cross_val[0][2] 
#                     best_score_val = sorted_auc_cross_val[0][0]

#                     auc_grid.append((best_score_val, best_nb_classifiers_val, best_tolerance_val))  
            
#             sorted_auc_grid = sorted(auc_grid, key=lambda tup:tup[0])[::-1]
                                                       
#             best_nb_classifiers = sorted_auc_grid[0][1]
#             best_tolerance = sorted_auc_grid[0][2]
                                      

                    
# #            print("nb_classifiers:", best_nb_classifiers, ", tolerance:", best_tolerance)
                    

#             model = HBagging()
#             model.fit(sub_X_train, y_train, nb_classifiers=best_nb_classifiers, tolerance=best_tolerance)      

#             selected_models[variable_type].append(model)

    
#             prediction_subtest = model.predict_proba(sub_X_test)[:,1]
#             predictions_subtest.append(prediction_subtest)

#             auc_score = roc_auc_score(y_true=y_test, y_score=prediction_subtest)
#             auc_subtest.append(auc_score)
            
#             print("AUC", auc_score)
                                
#             auc_variable_types[variable_type].append(auc_score)

#             variables_used = model.features_used
            

#             for i, var in enumerate(variables_names[index_list_]):
#                 dic_importances[var].append(1 if i in variables_used else 0)
    
    
# #    
# #            if plot:
# #                model.draw(sub_X_train, y_train, title="training_"+str(k)+variable_type, features_names=variables_names_clean[index_list_], rename_dic=rename_dic, fig_path=fig_path, reversed_variables_names_clean=reversed_variables_names)
# #                model.draw(sub_X_test, y_test, title="testing_"+str(k)+variable_type, features_names=variables_names_clean[index_list_], rename_dic=rename_dic, fig_path=fig_path, reversed_variables_names_clean=reversed_variables_names)
# #


#         predictions_subtest = np.array(predictions_subtest).T

#         final_predictions = np.mean(predictions_subtest, axis=1)

#         auc_score = roc_auc_score(y_true=y_test, y_score=final_predictions)
#         auc_test.append(auc_score)

            
#         fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=final_predictions)
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)

    
#     for variable_type, names_ in dic_variables.items():

#         importances = [np.mean(dic_importances[var]) for var in names_]
        
#         print("--")
#         print(variable_type)
#         print("--")
        
#         indices = np.argsort(np.abs(importances))[::-1]
#         sorted_variables = np.array(names_)[indices]
    
#         print("Poids des variables dans le modèle :\n", variable_type)

    
#         for i, var in enumerate(sorted_variables):
#             variable_list_importances = dic_importances[var]
#     #        if np.abs(np.mean(variables_list_importances)) < 0.05*np.max(np.abs(mean_importances)):
#     #            continue
#             if np.abs(np.mean(variable_list_importances)) == 0:
#                 continue
#             print(var, ", poids moyen :", "%.2f"%np.mean(variable_list_importances), \
#                           ", écart-type :", "%.2f"%np.std(variable_list_importances))
        
#     print("\nAUC moyenne sur les ensembles de test :", "%.3f"%np.mean(auc_test), "écart-type", "%.3f"%np.std(auc_test))

#     for variable_type in dic_variables:
            
#         aucs = auc_variable_types[variable_type]
#         print("\n", variable_type, len(index_list_), "%.3f"%np.mean(aucs), "écart-type", "%.3f"%np.std(aucs))
        

#     ax.set_ylabel("True positive rate", fontsize=30)
#     ax.set_xlabel("False positive rate", fontsize=30)


#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
    
#     std_auc = np.std(aucs)
#     ax.plot(mean_fpr, mean_tpr, color='b',
#             label=r'Mean ROC curve (AUC = %.2f $\pm$ %.2f)' % (np.mean(auc_test), np.std(auc_test)),
#             lw=2, alpha=.8)
    
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightgrey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')
    
#     ax.set_aspect("equal")

#     ax.legend(loc="lower right", fontsize=26) 
    
    
#     # Final model

#     dic_importances = {}
    
#     final_models = {}
    
#     variables_models = {}
    
    
#     print("final")

#     for variable_type, list_ in dic_variables.items():

#         index_list_ = []
#         for var in list_:
#             ind = list(variables_names).index(var)
#             index_list_.append(ind)
        
#         index_list_ = np.array(index_list_)

#         variables_models[variable_type] = index_list_

#         index_list_open = []
#         index_list_closed = []
        

    
#         sub_X = X[:,index_list_]
        
#         nb_classifiers_total = int(np.mean([model.nb_classifiers for model in selected_models[variable_type]]))
#         tolerance_total = np.mean([model.tolerance for model in selected_models[variable_type]])
        
#         print(variable_type, nb_classifiers_total, tolerance_total)
        
#         total_model = HBagging()
    
#         total_model.fit(sub_X, y, nb_classifiers=nb_classifiers_total, tolerance=tolerance_total)

        
#         variables_used = total_model.features_used

#         for i, var in enumerate(variables_names[index_list_]):
#             dic_importances[var] = 1 if i in variables_used else 0
    
#         final_models[variable_type] = total_model



        


#         if plot:
#             total_model.draw(sub_X, y, title="Hbagging_"+title+"_"+variable_type, features_names=variables_names[index_list_], rename_dic=rename_dic, fig_path=fig_path, reversed_variables_names=reversed_variables_names, \
#                              name_classes=name_classes)




#     for variable_type, names_ in dic_variables.items():

#         importances = [np.mean(dic_importances[var]) for var in names_]
        
#         print("--")
#         print(variable_type)
#         print("--")
        
#         indices = np.argsort(np.abs(importances))[::-1]
#         sorted_variables = np.array(names_)[indices]
    
#         print("Poids des variables dans le modèle final :\n")

    
#         for i, var in enumerate(sorted_variables):
#             variable_importances = dic_importances[var]
#     #        if np.abs(np.mean(variables_list_importances)) < 0.05*np.max(np.abs(mean_importances)):
#     #            continue
#             if variable_importances == 0:
#                 continue
#             print(var)

#     ax.tick_params(axis="both",labelsize=20)
#     fig.tight_layout()            
#     fig.savefig(Path(fig_path, "HBagging_roc_curve_"+title+".png"))

#     return final_models, variables_models









# def random_forest_classification_ikky_LOO(X, y, leave_one_out_values, params_grid, train_prop=0.8, \
#                                  num_iterations=10, cross_validations_splits=10, \
#                                  variable_names=None, stratify_=True, plot=True,
#                                  threshold=0.1):


    
    
#     auc_test = []
#     selected_models = []


#     strat = y if stratify_ else None

#     if variable_names is None:
#         variable_names = ["var "+str(i) for i in range(X.shape[1])]
        
#     dic_importances = {var:[] for var in variable_names}


    
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)


#     fig, ax = plt.subplots(1)
    
#     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#         label='Random classifier', alpha=.8)

#     ax.set_xlim([-0.05, 1.05])
#     ax.set_ylim([-0.05, 1.05])

#     unique_leave_one_out_values = np.unique(leave_one_out_values)
           
#     for pilot_to_leave_out in unique_leave_one_out_values:

#         test_pilot = pilot_to_leave_out
#         train_pilots = [k for k in unique_leave_one_out_values if k!=pilot_to_leave_out]
        
#         index_train = np.isin(leave_one_out_values, train_pilots)
#         index_test = leave_one_out_values == test_pilot
        
#         print("train on pilots", train_pilots, "test on pilot", pilot_to_leave_out)
        
#         X_train = X[index_train]
#         X_test = X[index_test]
#         y_train = y[index_train]
#         y_test = y[index_test]

            
#         grid = GridSearchCV(RandomForestClassifier(), params_grid, \
#                             cv=cross_validations_splits, scoring="roc_auc")
        
#         grid.fit(X_train, y_train)
    
#         keys_attributes = grid.cv_results_.keys()
        
#         df_results = pandas.DataFrame.from_dict(grid.cv_results_)
#         #print(df_results.columns)
#         #print(df_results[["params", "mean_test_score", "std_test_score"]])
    
#         model = grid.best_estimator_
#         selected_models.append(model)

#         auc_score = roc_auc_score(y_true=y_test, y_score=model.predict_proba(X_test)[:,1])
#         auc_test.append(auc_score)
        
#         print("AUC", auc_score)
        
#         for param in params_grid:
#             print(param, getattr(model,param))
        

#         importances = model.feature_importances_
        
#         for i, var in enumerate(variable_names):
#             dic_importances[var].append(importances[i])




#         fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=model.predict_proba(X_test)[:,1])
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)

        

#     mean_importances = [np.mean(dic_importances[var]) for var in variable_names]
#     indices = np.argsort(np.abs(mean_importances))[::-1]
#     sorted_variables = np.array(variable_names)[indices]


#     print("Poids des variables dans le modèle :\n")

#     for i, var in enumerate(sorted_variables):
#         variables_list_importances = dic_importances[var]
#         if np.abs(np.mean(variables_list_importances)) <= threshold*np.max(np.abs(mean_importances)):
#             continue
#         print(var, ", poids moyen :", "%.3f"%np.mean(variables_list_importances), \
#                       ", écart-type :", "%.3f"%np.std(variables_list_importances))
    

        
#     print("AUC moyenne sur les ensembles de test :", "%.2f"%np.mean(auc_test), "écart-type", "%.2f"%np.std(auc_test))





#     ax.set_ylabel("True positive rate", fontsize=25)
#     ax.set_xlabel("False positive rate", fontsize=25)


#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
    
#     std_auc = np.std(aucs)
#     ax.plot(mean_fpr, mean_tpr, color='b',
#             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#             lw=2, alpha=.8)
    
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightgrey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')
    
#     ax.set_aspect("equal")

#     ax.legend(loc="lower right", fontsize=20) 




# def logistic_lasso_classification_ikky_LOO(X, y, leave_one_out_values, params_grid, train_prop=0.8, \
#                                  num_iterations=10, cross_validations_splits=10, \
#                                  variable_names=None, stratify_=True, plot=True,
#                                  threshold=0.1):

    

#     auc_test = []
#     selected_models = []

#     strat = y if stratify_ else None

#     if variable_names is None:
#         variable_names = ["var "+str(i) for i in range(X.shape[1])]
        
#     dic_importances = {var:[] for var in variable_names}


    
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)


#     fig, ax = plt.subplots(1)
    
#     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#         label='Random classifier', alpha=.8)

#     ax.set_xlim([-0.05, 1.05])
#     ax.set_ylim([-0.05, 1.05])

    
#     unique_leave_one_out_values = np.unique(leave_one_out_values)
           
#     for pilot_to_leave_out in unique_leave_one_out_values:

#         test_pilot = pilot_to_leave_out
#         train_pilots = [k for k in unique_leave_one_out_values if k!=pilot_to_leave_out]
        
#         index_train = np.isin(leave_one_out_values, train_pilots)
#         index_test = leave_one_out_values == test_pilot
        
#         print("train on pilots", train_pilots, "test on pilot", pilot_to_leave_out)
        
#         X_train = X[index_train]
#         X_test = X[index_test]
#         y_train = y[index_train]
#         y_test = y[index_test]

            
#         # define min max scaler
#         scaler = MinMaxScaler()
#         # transform data
#         scaler.fit(X_train)
#         X_train = scaler.transform(X_train)
#         X_test = scaler.transform(X_test)




        



#         auc_score = roc_auc_score(y_true=y_test, y_score=model.predict_proba(X_test)[:,1])
        
#         auc_test.append(auc_score)

        
#         print("AUC", auc_score)
        
        
#         for param in params_grid:
#             print(param, getattr(model, param))
   
#         importances = model.coef_[0]
    
#         for i, var in enumerate(variable_names):
#             dic_importances[var].append(importances[i])


#         fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=model.predict_proba(X_test)[:,1])
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)


#     mean_importances = [np.mean(dic_importances[var]) for var in variable_names]
#     indices = np.argsort(np.abs(mean_importances))[::-1]
#     sorted_variables = np.array(variable_names)[indices]

#     print("Poids des variables dans le modèle :\n")

#     for i, var in enumerate(sorted_variables):
#         variables_list_importances = dic_importances[var]
#         if np.abs(np.mean(variables_list_importances)) <= threshold*np.max(np.abs(mean_importances)):
#             continue
#         print(var, ", poids moyen :", "%.3f"%np.mean(variables_list_importances), \
#                       ", écart-type :", "%.3f"%np.std(variables_list_importances))
    

        
#     print("AUC moyenne sur les ensembles de test :", "%.2f"%np.mean(auc_test), "écart-type", "%.2f"%np.std(auc_test))





#     ax.set_ylabel("True positive rate", fontsize=25)
#     ax.set_xlabel("False positive rate", fontsize=25)





#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
    

#     ax.plot(mean_fpr, mean_tpr, color='b',
#             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (np.mean(auc_test), np.std(auc_test)),
#             lw=2, alpha=.8)
    
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='lightgrey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')
    
#     ax.set_aspect("equal")

#     ax.legend(loc="lower right", fontsize=20) 

