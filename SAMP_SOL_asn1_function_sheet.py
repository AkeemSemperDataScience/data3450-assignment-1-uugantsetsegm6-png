
import pandas as pd
import numpy as np
import math

def age_splitter(df, col_name, age_threshold):
    """
    Splits the dataframe into two dataframes based on an age threshold.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    col_name (str): The name of the column containing age values.
    age_threshold (int): The age threshold for splitting.

    Returns:
    tuple: A tuple containing two dataframes:
        - df_below: DataFrame with rows where age is below the threshold.
        - df_above_equal: DataFrame with rows where age is above or equal to the threshold.
    """
    df_below = df[df[col_name] < age_threshold]
    df_above_equal = df[df[col_name] >= age_threshold]
    
    return df_below, df_above_equal

def effectSizer(df, num_col, cat_col):
    """
    Calculates the effect sizes of binary categorical classes on a numerical value.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    num_col (str): The name of the numerical column.
    cat_col (str): The name of the binary categorical column.

    Returns:
    float: Cohen's d effect size between the two groups defined by the categorical column.
    Raises:
    ValueError: If the categorical column does not have exactly two unique values.
    """

    possible_options = df[cat_col].unique()
    if len(possible_options) != 2:
        raise ValueError("The categorical column must have exactly two unique values.")
   
    # Calculate cohens d
    group1 = df[df[cat_col] == possible_options[0]][num_col]
    group2 = df[df[cat_col] == possible_options[1]][num_col]
    d = cohenEffectSize(group1, group2)
    return d

def cohenEffectSize(group1, group2):
    diff = group1.mean() - group2.mean()
    var1 = group1.var()
    var2 = group2.var()
    n1 = len(group1)
    n2 = len(group2)
    pooled_var = math.sqrt((n1 * var1 + n2 * var2) / (n1 + n2))
    d = diff/math.sqrt(pooled_var)
    return d

def cohortCompare(df, cohorts, statistics=['mean', 'median', 'std', 'min', 'max']):

    cohort_metrics = {}

    for cohort in cohorts:
        categorical_values = df[cohort].unique()
        for value in categorical_values:
            cohort_name = f"{cohort}_{value}"
            cohort_df = df[df[cohort] == value]
            metric = CohortMetric(cohort_name)

            if 'mean' in statistics:
                metric.setMean(cohort_df.select_dtypes(include=np.number).mean())
            if 'median' in statistics:
                metric.setMedian(cohort_df.select_dtypes(include=np.number).median())
            if 'std' in statistics:
                metric.setStd(cohort_df.select_dtypes(include=np.number).std())
            if 'min' in statistics:
                metric.setMin(cohort_df.select_dtypes(include=np.number).min())
            if 'max' in statistics:
                metric.setMax(cohort_df.select_dtypes(include=np.number).max())

            cohort_metrics[cohort_name] = metric
            #print(metric)
    #print(cohort_metrics)
    return cohort_metrics

class CohortMetric():
    def __init__(self, cohort_name):
        self.cohort_name = cohort_name
        self.statistics = {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None
        }
    def setMean(self, new_mean):
        self.statistics["mean"] = new_mean
    def setMedian(self, new_median):
        self.statistics["median"] = new_median
    def setStd(self, new_std):
        self.statistics["std"] = new_std
    def setMin(self, new_min):
        self.statistics["min"] = new_min
    def setMax(self, new_max):
        self.statistics["max"] = new_max

    def compare_to(self, other):
        for stat in self.statistics:
            if not self.statistics[stat].equals(other.statistics[stat]):
                return False
        return True
    def __str__(self):
        output_string = f"\nCohort: {self.cohort_name}\n"
        for stat, value in self.statistics.items():
            output_string += f"\t{stat}:\n{value}\n"
            output_string += "\n"
        return output_string