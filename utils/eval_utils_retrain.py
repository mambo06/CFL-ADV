import csv
import functools
import os

import numpy as np
import pandas as pd
import torch as th
import xgboost as xgb

from sklearn.metrics import mean_squared_error, precision_recall_fscore_support


def linear_model_eval(
    config,
    z_train,
    y_train,
    suffix,
    z_test,
    y_test,
    z_val,
    y_val,
    description="XGBoost Evaluation",
    models=None,
    x_=None,
    y_=None,
):
    results_list = []

    print(10 * ">" + description)

    prefix = str(config["prefix"]) + suffix
    file_name = prefix

    param_grid = {
        "max_depth": [8, 10],
        "n_estimators": [900, 1000],
        "learning_rate": [0.01, 0.015],
    }

    regularisation_list = [10]

    for c in regularisation_list:
        print(10 * "*" + "parameters=" + str(c) + 10 * "*")

        if config["task_type"] == "regression":
            clf = xgb.XGBRegressor(
                learning_rate=param_grid["learning_rate"][-1],
                n_estimators=param_grid["n_estimators"][-1],
                max_depth=param_grid["max_depth"][-1],
                colsample_bytree=config.get("colsample_bytree", 1.0),
                subsample=config.get("subsample", 1.0),
                objective="reg:squarederror",
                eval_metric="rmse",
                verbosity=0,
            )

            if models is None:
                clf.fit(z_train, y_train)

            elif suffix.split("-")[1] == "ClNoRetrain":
                clf.fit(
                    z_test,
                    y_test,
                    verbose=0,
                    eval_set=[(z_val, y_val)],
                )

            else:
                clf.fit(
                    z_test,
                    y_test,
                    xgb_model=models,
                    verbose=0,
                    eval_set=[(z_val, y_val)],
                )

            y_hat_train = clf.predict(z_train)
            y_hat_test = clf.predict(z_test)
            y_hat_val = clf.predict(z_val)

            tr_acc = np.sqrt(mean_squared_error(y_train, y_hat_train))
            te_acc = np.sqrt(mean_squared_error(y_test, y_hat_test))
            ve_acc = np.sqrt(mean_squared_error(y_val, y_hat_val))

            print("Training RMSE:   {}".format(tr_acc))
            print("Validation RMSE: {}".format(ve_acc))
            print("Test RMSE:       {}".format(te_acc))

            results_list.append(
                {
                    "model": "XGBRegressor_" + str(c),
                    "train_acc": tr_acc,
                    "test_acc": te_acc,
                    "val_acc": ve_acc,
                }
            )

        else:
            clf = xgb.XGBClassifier(
                learning_rate=param_grid["learning_rate"][-1],
                n_estimators=param_grid["n_estimators"][-1],
                max_depth=param_grid["max_depth"][-1],
                colsample_bytree=config.get("colsample_bytree", 1.0),
                subsample=config.get("subsample", 1.0),
                eval_metric="mlogloss",
                verbosity=0,
            )

            if models is None:
                clf.fit(z_train, y_train)

            elif suffix.split("-")[1] == "ClNoRetrain":
                if x_ is not None and y_ is not None:
                    clf.fit(x_, y_)
                else:
                    clf.fit(z_test, y_test)

            else:
                if x_ is not None and y_ is not None:
                    clf.fit(
                        x_,
                        y_,
                        xgb_model=models,
                    )
                else:
                    clf.fit(
                        z_test,
                        y_test,
                        xgb_model=models,
                    )

            y_hat_train = clf.predict(z_train)
            y_hat_test = clf.predict(z_test)
            y_hat_val = clf.predict(z_val)

            tr_acc = precision_recall_fscore_support(
                y_train,
                y_hat_train,
                average="macro",
                zero_division=0,
            )

            val_acc = precision_recall_fscore_support(
                y_val,
                y_hat_val,
                average="macro",
                zero_division=0,
            )

            te_acc = precision_recall_fscore_support(
                y_test,
                y_hat_test,
                average="macro",
                zero_division=0,
            )

            print(
                "Training score: precision {}, recall {}, F1 {}, support {}".format(
                    tr_acc[0],
                    tr_acc[1],
                    tr_acc[2],
                    tr_acc[3],
                )
            )

            print(
                "Validation score: precision {}, recall {}, F1 {}, support {}".format(
                    val_acc[0],
                    val_acc[1],
                    val_acc[2],
                    val_acc[3],
                )
            )

            print(
                "Test score: precision {}, recall {}, F1 {}, support {}".format(
                    te_acc[0],
                    te_acc[1],
                    te_acc[2],
                    te_acc[3],
                )
            )

            results_list.append(
                {
                    "model": "XGBClassifier_" + str(c), #LogReg_
                    "train_acc": tr_acc,
                    "test_acc": te_acc,
                    "val_acc": val_acc,
                }
            )

    keys = results_list[0].keys()

    file_path = "./results/" + config["dataset"] + "/" + file_name + ".csv"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_list)

        print(f"{100 * '='}\n")
        print(f"Training results are saved at: {file_path}")

    if models is None:
        return clf

    return results_list


def save_np2csv(np_list, save_as="test.csv"):
    """
    Saves a list of numpy arrays to a CSV file.

    Args:
        np_list (list): List containing features and labels.
        save_as (str): File name to save as.
    """

    Xtr, ytr = np_list

    ytr = np.array(ytr, dtype=np.int8)

    columns = ["label"] + list(map(str, list(range(Xtr.shape[1]))))

    data_tr = np.concatenate((ytr.reshape(-1, 1), Xtr), axis=1)

    df_tr = pd.DataFrame(data=data_tr, columns=columns)

    print("Samples from the dataframe:")
    print(df_tr.head())

    df_tr.to_csv(save_as, index=False)

    print(f"The dataframe is saved as {save_as}")


def append_tensors_to_lists(list_of_lists, list_of_tensors):
    """
    Appends tensors in a list to a list after converting tensors to numpy arrays.

    Args:
        list_of_lists (list): List of lists, each holding arrays.
        list_of_tensors (list): List of PyTorch tensors.

    Returns:
        list: Updated list of lists.
    """

    for i in range(len(list_of_tensors)):
        list_of_lists[i] += [list_of_tensors[i].detach().numpy()]

    return list_of_lists


def concatenate_lists(list_of_lists):
    """
    Concatenates each list with the main list to a numpy array.

    Args:
        list_of_lists (list): List of lists containing numpy arrays.

    Returns:
        numpy.ndarray or list: Concatenated numpy arrays.
    """

    list_of_np_arrs = []

    for list_ in list_of_lists:
        list_of_np_arrs.append(np.concatenate(list_))

    return list_of_np_arrs[0] if len(list_of_np_arrs) == 1 else list_of_np_arrs


def aggregate(latent_list, config):
    """
    Aggregates latent representations of subsets to obtain joint representation.

    Args:
        latent_list (list): List of latent variables.
        config (dict): Dictionary holding the configuration.

    Returns:
        torch.Tensor: Joint representation.
    """

    latent = None

    if config["aggregation"] == "mean":
        latent = sum(latent_list) / len(latent_list)

    elif config["aggregation"] == "sum":
        latent = sum(latent_list)

    elif config["aggregation"] == "concat":
        latent = th.cat(latent_list, dim=-1)

    elif config["aggregation"] == "max":
        latent = functools.reduce(th.max, latent_list)

    elif config["aggregation"] == "min":
        latent = functools.reduce(th.min, latent_list)

    else:
        print("Proper aggregation option is not provided. Please check the config file.")
        exit()

    return latent
