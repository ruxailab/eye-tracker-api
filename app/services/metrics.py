import numpy as np


def func_precision_x(group):
    """
    Calculate the precision for the X axis.

    Args:
        group (pandas.DataFrame): A group of data containing the predicted and true values for the X axis.

    Returns:
        float: The precision value.
    """
    return np.sqrt(
        np.mean(np.square(group["Predicted X"] - np.mean(group["Predicted X"])))
    )

def func_presicion_y(group):
    """
    Calculate the precision for the Y axis.

    Args:
        group (pandas.DataFrame): A group of data containing the predicted and true values for the Y axis.

    Returns:
        float: The precision value.
    """
    return np.sqrt(
        np.mean(np.square(group["Predicted Y"] - np.mean(group["Predicted Y"])))
    )


def func_accuracy_x(group):
    """
    Calculate the accuracy for the X axis.

    Args:
        group (pandas.DataFrame): A group of data containing the predicted and true values for the X axis.

    Returns:
        float: The accuracy value.
    """
    
    return np.sqrt(np.mean(np.square(group["True X"] - group["Predicted X"])))



def func_accuracy_y(group):
    """
    Calculate the accuracy for the Y axis.

    Args:
        group (pandas.DataFrame): A group of data containing the predicted and true values for the Y axis.

    Returns:
        float: The accuracy value.
    """
    return np.sqrt(np.mean(np.square(group["True Y"] - group["Predicted Y"])))

