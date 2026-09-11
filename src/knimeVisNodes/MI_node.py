import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import knime.extension as knext

LOGGER = logging.getLogger(__name__)

from utils import knutils as kutil
knimeVis_category = kutil.get_knimeVis_category ()

@knext.node(
    name="Mutual Information Filter",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/MI-filter.png",
    category=knimeVis_category,
    id="mi-filter"
)
@knext.input_table(
    name="Input table",
    description="Table containing numeric feature columns and a class label column.",
)
@knext.output_table(
    name="Ranked features",
    description="One row per feature: mean/std MI over trials, selection frequency, "
                "rank, and top-k selection flag, sorted descending by mean MI.",
)
class MIFeatureRankerSklearnNode:
    """Ranks feature columns by mutual_info_classif MI against a class column,
    averaged over multiple seeded trials to both stabilize the score and report
    how stable the top-k selection itself."""

    class_column = knext.ColumnParameter(
        label="Class column",
        description="Column holding the numeric class label.",
        column_filter=lambda c: True,
    )

    allow_string_class = knext.BoolParameter(
        label="Allow string class column",
        description="If enabled, a string class column is accepted and encoded internally "
                     "mapping written to the node's log. Prefer encoding the class "
                     "column explicitly upstream instead",
        default_value=False,
    )
    
    feature_columns = knext.MultiColumnParameter(
        label="Feature columns",
        description="Numeric columns to rank against the class column ",
        column_filter=lambda c: c.ktype == knext.double() or c.ktype == knext.int32(),
    )

    trials = knext.IntParameter(
        label="Trials",
        description="Number of seeded mutual_info_classif runs per feature, matching "
                     "the original script's `trials = 10`. Used to average the MI score "
                     "and to measure how stable the top-k selection is across the "
                     "estimator's internal randomness.",
        default_value=10,
        min_value=1,
    )

    base_seed = knext.IntParameter(
        label="seed",
        description="Trial i uses random_state = base_seed + i. Fixing this set of seeds "
                     "makes the node's output identical across executions, while still exercising the "
                     "estimator's randomness across `trials` different draws.",
        default_value=0,
    )

    n_neighbors = knext.IntParameter(
        label="Number of neighbors",
        description="k for the KNN-based estimator. sklearn default is 3.",
        default_value=3,
        min_value=1,
    )

    top_k_percent = knext.DoubleParameter(
        label="Top-k percent feature selection",
        description="Percentage (0-100) of features to flag as 'Selected'.",
        default_value=10.0,
        min_value=0.0,
        max_value=100.0,
    )

    def configure(self, config_context, input_schema):
        if self.class_column is None:
            raise knext.InvalidParametersError("Select a class column.")
        if not self.feature_columns:
            raise knext.InvalidParametersError("Select at least one feature column.")
        
        class_ktype = input_schema[self.class_column].ktype
        is_numeric = class_ktype in (knext.int32(), knext.int64(), knext.double())
        if not is_numeric and not self.allow_string_class:
            raise knext.InvalidParametersError(
                f"Class column '{self.class_column}' is not numeric (type: {class_ktype}). "
                "Encode it to 0/1 upstream so the mapping is explicit in the workflow, or enable 'Allow string class column' "
                "to have this node encode it internally."
            )
        return knext.Schema.from_columns([
            knext.Column(knext.string(), "Feature"),
            knext.Column(knext.double(), "MeanMI"),
            knext.Column(knext.double(), "StdMI"),
            knext.Column(knext.double(), "SelectionFrequency"),
            knext.Column(knext.int32(), "Rank"),
            knext.Column(knext.bool_(), "Selected"),
        ])

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        y_raw = df[self.class_column]
 
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.to_numpy()
        else:
            classes, y = np.unique(y_raw.to_numpy(), return_inverse=True)
            mapping = {str(c): i for i, c in enumerate(classes)}
            LOGGER.warning(
                f"Class column '{self.class_column}' was a string column and has been "
                f"auto-encoded (alphabetical order): {mapping}. Consider encoding this "
                f"upstream instead so the mapping is visible in the workflow."
            )
        n_features = len(self.feature_columns)
        n_trials = self.trials

        # mi_matrix[trial, feature]
        mi_matrix = np.zeros((n_trials, n_features))

        for t in range(n_trials):
            seed = self.base_seed + t
            for i, col in enumerate(self.feature_columns):
                x_col = df[[col]].to_numpy()
                mi = mutual_info_classif(
                    x_col, y,
                    discrete_features="auto",   # ROI features are continuous, no auto-guessing and set to false
                    n_neighbors=self.n_neighbors,
                    random_state=seed,
                )
                mi_matrix[t, i] = mi[0]
            exec_context.set_progress((t + 1) / n_trials)

        mean_mi = mi_matrix.mean(axis=0)
        std_mi = mi_matrix.std(axis=0)

        # Per-trial top-k membership, using the sort-and-midpoint threshold
        # applied independently within each trial.
        max_ff = max(1, int(np.ceil(self.top_k_percent / 100.0 * n_features)))
        membership = np.zeros((n_trials, n_features), dtype=bool)
        for t in range(n_trials):
            trial_vals = mi_matrix[t]
            sorted_vals = np.sort(trial_vals)[::-1]
            if max_ff < n_features:
                thresh_t = (sorted_vals[max_ff] + sorted_vals[max_ff - 1]) / 2.0
            else:
                thresh_t = sorted_vals[-1]
            membership[t] = trial_vals >= thresh_t
        selection_frequency = membership.mean(axis=0)

        result_df = pd.DataFrame({
            "Feature": self.feature_columns,
            "MeanMI": mean_mi,
            "StdMI": std_mi,
            "SelectionFrequency": selection_frequency,
        })

        # Final selection: top-k% by MeanMI
        sorted_mean = np.sort(result_df["MeanMI"].to_numpy())[::-1]
        if max_ff < n_features:
            mi_threshold = (sorted_mean[max_ff] + sorted_mean[max_ff - 1]) / 2.0
        else:
            mi_threshold = sorted_mean[-1]

        result_df = result_df.sort_values("MeanMI", ascending=False).reset_index(drop=True)
        result_df["Rank"] = np.arange(1, len(result_df) + 1)
        result_df["Selected"] = result_df["MeanMI"] >= mi_threshold

        n_selected = int(result_df["Selected"].sum())
        unstable = result_df[(result_df["Selected"]) & (result_df["SelectionFrequency"] < 1.0)]
        LOGGER.info(
            f"MI Feature Ranker (sklearn): {n_trials} trials (seeds {self.base_seed}.."
            f"{self.base_seed + n_trials - 1}), threshold={mi_threshold:.6f}, "
            f"{n_selected} of {n_features} features selected at top {self.top_k_percent}%. "
            f"{len(unstable)} selected feature(s) had SelectionFrequency < 1.0 "
            f"(selection sensitive to which trial's draw is used)."
        )

        return knext.Table.from_pandas(result_df)
    
