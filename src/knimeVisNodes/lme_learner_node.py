import logging
import knime.extension as knext
LOGGER = logging.getLogger(__name__)
from utils import knutils as kutil
knimeVis_category = kutil.get_knimeVis_category ()
from lme_main.lme_port import (  
    LMEModelSpec,
    LMEModelPortObject,
    lme_model_port_type,
)

@knext.node(
    name="LME Modeling (Learner)",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/lme-learner.png",
    category=knimeVis_category,
    id="lme-learner"
)
@knext.input_table(name="Training data", description="Longitudinal data, long format: one row per subject visit")
@knext.output_port(name = "LME Model", description= "Fitted fixed effects per target, for use with LME Modeling (Apply)", port_type = lme_model_port_type)
@knext.output_table(name="Training features", description="One row per subject, target-prefixed LME-derived features")
class LMEModelingLearner(knext.PythonNode):

    subject_id_col = knext.ColumnParameter(
        "Subject / group ID column",
        "Column identifying which rows belong to the same subject across visits.",
        port_index=0,
    )

    time_col = knext.ColumnParameter(
        "Time / interaction column",
        "The longitudinal timeline variable (e.g. age, months since baseline, "
        "visit number). This is the sole covariate fit in v1 — dynamic, not "
        "hardcoded to 'age'.",
        port_index=0,
        
    )

    target_cols = knext.MultiColumnParameter(
        "Target / ROI columns",
        "One or more numeric columns to fit a separate linear LME against. "
        "Output is pivoted to one row per subject, with columns prefixed "
        "per target (e.g. FS383_dev_bl).",
        port_index=0,
        
    )

    passthrough_cols = knext.MultiColumnParameter(
        "Passthrough columns (optional)",
        "Static columns to carry into the output unchanged, taken from each "
        "subject's last visit (e.g. SEX).",
        port_index=0,
        since_version="1.0.0",
    )

    def configure(self, ctx: knext.ConfigurationContext, input_schema: knext.Schema):
        from lme_main.lme_port import LMEModelSpec

        if self.subject_id_col is None or self.time_col is None or not self.target_cols:
            raise knext.InvalidParametersError(
                "Select a subject ID column, a time column, and at least one target column."
            )

        spec = LMEModelSpec(
            id_col=self.subject_id_col,
            time_col=self.time_col,
            targets=list(self.target_cols),
            train_time_min=None,
            train_time_max=None,
        )

        # Return empty schema for the table output — full wide schema is
        # only known after fitting, resolved at execute() time
        return spec, knext.Schema([], [])
    def execute(self, ctx: knext.ExecutionContext, input_table: knext.Table):
        import pandas as pd
        from lme_main.lme_port import LMEModelSpec, LMEModelPortObject 
        from lme_main import lme_core 
        df = input_table.to_pandas()
        id_col, time_col = self.subject_id_col, self.time_col
        targets = list(self.target_cols)

        df = lme_core.validate_and_clean(df, id_col, time_col, targets)

        fitted_models: dict[str, lme_core.FittedTargetModel] = {}
        per_target_frames = []

        for i, target in enumerate(targets):
            ctx.set_progress(i / len(targets), f"Fitting {target}")
            fitted, result = lme_core.fit_linear_lme(df, id_col, time_col, target)
            fitted_models[target] = fitted

            det = lme_core.extract_deterministic_features(df, id_col, time_col, target, fitted)
            re = lme_core.measured_random_effects(id_col, result)

            rows = []
            for sid in det:
                row = {id_col: sid}
                for suffix in lme_core.PER_TARGET_SUFFIXES:
                    value = re[sid][suffix] if suffix in ("ra_int", "ra_slope") else det[sid][suffix]
                    row[lme_core.target_col(target, suffix)] = value
                rows.append(row)
            target_df = pd.DataFrame(rows).set_index(id_col)
            per_target_frames.append(target_df)

        # subject-level columns (age_mean / age_max) computed once, not per target
        subject_level = (
            df.groupby(id_col)[time_col]
            .agg(age_mean="mean", age_max="max")
            .astype(float)
        )
        subject_level.index = subject_level.index.astype(str)

        wide = subject_level
        for tdf in per_target_frames:
            wide = wide.join(tdf, how="outer")

        if self.passthrough_cols:
            last_visit = df.sort_values(time_col).groupby(id_col).last()
            last_visit.index = last_visit.index.astype(str)
            wide = wide.join(last_visit[list(self.passthrough_cols)], how="left")

        wide = wide.reset_index().rename(columns={"index": id_col})

        train_min = float(df[time_col].min())
        train_max = float(df[time_col].max())
        out_spec = LMEModelSpec(id_col, time_col, targets, train_min, train_max)
        model_port = LMEModelPortObject(out_spec, fitted_models)

        return model_port, knext.Table.from_pandas(wide)
