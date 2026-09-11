#from __future__ import annotations
import logging
import knime.extension as knext
from lme_main.lme_port import (  
    LMEModelSpec,
    LMEModelPortObject,
    lme_model_port_type,
)

#from lme_main import lme_model_port_type
#from lme_main.lme_port import LMEModelSpec, LMEModelPortObject

LOGGER = logging.getLogger(__name__)

from utils import knutils as kutil
knimeVis_category = kutil.get_knimeVis_category ()


@knext.node(
    name="LME Modeling (Apply)",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/lme-apply.png",
    category="knimeVis_category",
    id= "lme-apply"
)
@knext.input_port(name="LME Model", description="Fitted model from LME Modeling (Learner)", port_type=lme_model_port_type)
@knext.input_table(name="New data", description="Longitudinal data for new/held-out subjects, long format")
@knext.output_table(name="Scored features", description="Same target-prefixed schema as the Learner's training features")
class LMEModelingApply:

    id_col_override = knext.ColumnParameter(
        "Subject / group ID column (if different from training)",
        "Defaults to the ID column used at training time if left unset.",
        port_index=1,
        #optional=True,
    )

    time_col_override = knext.ColumnParameter(
        "Time / interaction column (if different from training)",
        "Defaults to the time column used at training time if left unset.",
        port_index=1,
        #optional=True,
        #column_filter=knext.ColumnFilter.include_type(knext.double(), knext.int_()),
    )

    def configure(self, ctx: knext.ConfigurationContext, model_spec: LMEModelSpec, input_schema: knext.Schema):
        id_col = self.id_col_override or model_spec.id_col
        time_col = self.time_col_override or model_spec.time_col
        for col in [id_col, time_col] + model_spec.targets:
            if col not in input_schema.column_names:
                raise knext.InvalidParametersError(
                    f"Column '{col}' required by the fitted model is missing "
                    f"from the new data table."
                )
        return knext.Schema([], []) # schema depends on wide pivot, resolved at execute time

    def execute(self, ctx, model_port, input_table):
        import pandas as pd
        from lme_main import lme_core

        spec     = model_port.spec
        targets  = spec.targets

        # Read id_col and time_col from FittedTargetModel — more reliable than
        # spec after deserialization, since FittedTargetModel is stored in bytes
        first_fitted = list(model_port.fitted_models.values())[0]
        id_col   = self.id_col_override  or first_fitted.id_col
        time_col = self.time_col_override or first_fitted.time_col

        df = input_table.to_pandas()
    
        df = lme_core.validate_and_clean(df, id_col, time_col, targets)

        train_min, train_max = spec.train_time_range
        warnings = lme_core.check_age_extrapolation(df, id_col, time_col, train_min, train_max)
        for w in warnings:
            ctx.set_warning(w)
        if warnings:
            LOGGER.warning(
                "%d subject(s) fall outside the training time range %.2f-%.2f",
                len(warnings), train_min, train_max,
            )

        per_target_frames = []
        for i, target in enumerate(targets):
            ctx.set_progress(i / len(targets), f"Scoring {target}")
            fitted = model_port.fitted_models[target]

            det = lme_core.extract_deterministic_features(df, id_col, time_col, target, fitted)
            eb = lme_core.empirical_bayes_random_effects(df, id_col, time_col, target, fitted)

            rows = []
            for sid in det:
                row = {id_col: sid}
                for suffix in lme_core.PER_TARGET_SUFFIXES:
                    value = eb[sid][suffix] if suffix in ("ra_int", "ra_slope") else det[sid][suffix]
                    row[lme_core.target_col(target, suffix)] = value
                rows.append(row)
            per_target_frames.append(pd.DataFrame(rows).set_index(id_col))

        subject_level = (
            df.groupby(id_col)[time_col]
            .agg(age_mean="mean", age_max="max")
            .astype(float)
        )
        subject_level.index = subject_level.index.astype(str)

        wide = subject_level
        for tdf in per_target_frames:
            wide = wide.join(tdf, how="outer")

        wide = wide.reset_index().rename(columns={"index": id_col})

        return knext.Table.from_pandas(wide)
