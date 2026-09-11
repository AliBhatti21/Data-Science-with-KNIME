import logging
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
import knime.extension as knext

LOGGER = logging.getLogger(__name__)

from utils import knutils as kutil
knimeVis_category = kutil.get_knimeVis_category ()

class ICVNormalizerModelSpec(knext.PortObjectSpec):
    def __init__(self, icv_column, feature_columns):
        self._icv_column = icv_column
        self._feature_columns = list(feature_columns)

    @property
    def icv_column(self):
        return self._icv_column

    @property
    def feature_columns(self):
        return self._feature_columns

    def serialize(self):
        return {"icv_column": self._icv_column, "feature_columns": self._feature_columns}

    @classmethod
    def deserialize(cls, data):
        return cls(data["icv_column"], data["feature_columns"])


class ICVNormalizerModel(knext.PortObject):
    """Fitted mean ICV and per-feature regression slopes (b-weights) learned
    on the Learner node's input table. The Apply node uses these unchanged --
    it never recomputes mean_icv or any b from the data it is given."""

    def __init__(self, spec, mean_icv, b_weights):
        super().__init__(spec)
        self._mean_icv = mean_icv
        self._b_weights = b_weights  # dict: feature_name -> slope (float)

    @property
    def mean_icv(self):
        return self._mean_icv

    @property
    def b_weights(self):
        return self._b_weights

    def serialize(self):
        return pickle.dumps({"mean_icv": self._mean_icv, "b_weights": self._b_weights})

    @classmethod
    def deserialize(cls, spec, data):
        payload = pickle.loads(data)
        return cls(spec, payload["mean_icv"], payload["b_weights"])


icv_model_port_type = knext.port_type(
    "ICV Normalizer Model", ICVNormalizerModel, ICVNormalizerModelSpec
)


def _drop_incomplete_rows(df, critical_cols, node_name):
    n_before = len(df)
    clean_mask = df[critical_cols].notna().all(axis=1)
    df_clean = df[clean_mask].copy()
    n_dropped = n_before - len(df_clean)
    if n_dropped > 0:
        LOGGER.warning(
            f"{node_name}: dropped {n_dropped} of {n_before} rows with missing "
            f"values in the ICV column or a selected feature column "
            f"(no imputation, matching the original script's behavior)."
        )
    return df_clean


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

@knext.node(
    name="ICV Normalizer (Learner)",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/icv_normalizer.png",
    category=knimeVis_category,
)
@knext.input_table(
    name="Training data",
    description="Training data containing the ICV column and the ROI feature columns to normalize.",
)
@knext.output_table(
    name="Normalized training data",
    description="Training data with the selected ROI features ICV-adjusted. Rows with "
                 "missing values in the ICV column or any selected feature column are dropped.",
)
@knext.output_port(
    name="ICV model",
    description="Fitted mean ICV and per-feature regression slopes, for the "
                 "ICV Normalizer (Apply) node to reuse on new data without refitting.",
    port_type=icv_model_port_type,
)
class ICVNormalizerLearnerNode:
    """Fits an ICV head-size correction (Buckner et al., 2004 residual method):
    for each selected ROI feature, regresses the feature on ICV to obtain a
    slope b, then adjusts each subject's value as
        adjusted = raw - b * (ICV_subject - mean_ICV_training)
    mean_ICV and each b are estimated only on this node's input table and are
    output as a model port for the Apply node -- never recomputed downstream."""

    icv_column = knext.ColumnParameter(
        label="Estimated Intracranial Volume Column",
        description="Column holding the estimated intracranial volume measurement (e.g. eTIV).",
        column_filter=lambda c: c.ktype == knext.double() or c.ktype == knext.int32(),
    )

    feature_columns = knext.MultiColumnParameter(
        label="Feature columns to normalize",
        description="ROI feature columns to ICV-adjust. This node does not infer which "
                     "measures should be excluded (e.g. cortical thickness, surface area, "
                     "curvature) -- select explicitly the columns you want corrected.",
        column_filter=lambda c: c.ktype == knext.double() or c.ktype == knext.int32(),
    )

    def configure(self, config_context, input_schema):
        if self.icv_column is None:
            raise knext.InvalidParametersError("Select the ICV column.")
        if not self.feature_columns:
            raise knext.InvalidParametersError("Select at least one feature column to normalize.")
        if self.icv_column in self.feature_columns:
            raise knext.InvalidParametersError(
                "The ICV column cannot also be selected as a feature column to normalize."
            )
        model_spec = ICVNormalizerModelSpec(self.icv_column, self.feature_columns)
        return input_schema, model_spec

    def execute(self, exec_context, input_table):
        df = input_table.to_pandas()
        critical_cols = [self.icv_column] + list(self.feature_columns)
        df_clean = _drop_incomplete_rows(df, critical_cols, "ICV Normalizer (Learner)")

        icv = df_clean[self.icv_column].to_numpy()
        mean_icv = float(icv.mean())

        b_weights = {}
        for col in self.feature_columns:
            X = icv.reshape(-1, 1)
            y_roi = df_clean[col].to_numpy()
            b = float(LinearRegression().fit(X, y_roi).coef_[0])
            b_weights[col] = b
            df_clean[col] = df_clean[col] - b * (icv - mean_icv)

        LOGGER.info(
            f"ICV Normalizer (Learner): fitted mean_icv={mean_icv:.4f} over "
            f"{len(df_clean)} rows; {len(b_weights)} feature slope(s) computed."
        )

        model_spec = ICVNormalizerModelSpec(self.icv_column, self.feature_columns)
        model = ICVNormalizerModel(model_spec, mean_icv, b_weights)
        return knext.Table.from_pandas(df_clean), model


# -----------
# Apply
# -----------

@knext.node(
    name="ICV Normalizer (Apply)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/icv-apply.png",
    category=knimeVis_category,
)
@knext.input_port(
    name="ICV model",
    description="Model produced by ICV Normalizer (Learner).",
    port_type=icv_model_port_type,
)
@knext.input_table(
    name="New data",
    description="Data to normalize with the fitted ICV parameters (e.g. the held-out test set). "
                 "Must contain the same ICV column and feature columns used to fit the model.",
)
@knext.output_table(
    name="Normalized data",
    description="Input data with the model's feature columns ICV-adjusted using the frozen "
                 "mean_icv and b_weights -- no refitting occurs in this node.",
)
class ICVNormalizerApplyNode:
    """Applies a previously fitted ICV correction to new data without recomputing
    mean_icv or any b -- the Learner/Apply split that prevents new-data statistics
    from leaking into the normalization parameters, mirroring KNIME's built-in
    Normalizer / Normalizer (Apply) pair."""

    def configure(self, config_context, model_spec, input_schema):
        required = [model_spec.icv_column] + list(model_spec.feature_columns)
        missing = [c for c in required if c not in input_schema.column_names]
        if missing:
            raise knext.InvalidParametersError(
                f"Input table is missing column(s) required by the fitted model: {missing}"
            )
        return input_schema

    def execute(self, exec_context, model, input_table):
        df = input_table.to_pandas()
        icv_column = model.spec.icv_column
        feature_columns = model.spec.feature_columns
        critical_cols = [icv_column] + list(feature_columns)
        df_clean = _drop_incomplete_rows(df, critical_cols, "ICV Normalizer (Apply)")

        icv = df_clean[icv_column].to_numpy()
        mean_icv = model.mean_icv  # frozen from the Learner node -- not recomputed here
        for col in feature_columns:
            b = model.b_weights[col]  # frozen from the Learner node -- not refit here
            df_clean[col] = df_clean[col] - b * (icv - mean_icv)

        LOGGER.info(
            f"ICV Normalizer (Apply): applied fitted mean_icv={mean_icv:.4f} to "
            f"{len(df_clean)} rows using {len(feature_columns)} feature slope(s) "
            f"(no refitting)."
        )

        return knext.Table.from_pandas(df_clean)