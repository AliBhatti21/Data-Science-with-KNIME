from __future__ import annotations
import json
import knime.extension as knext

class LMEModelSpec(knext.PortObjectSpec):
    def __init__(self, id_col: str, time_col: str, targets: list,
             train_time_min, train_time_max):
        self._id_col = id_col
        self._time_col = time_col
        self._targets = list(targets)
        self._train_time_min = train_time_min  # can be None or float
        self._train_time_max = train_time_max

    @property
    def id_col(self) -> str:
        return self._id_col

    @property
    def time_col(self) -> str:
        return self._time_col

    @property
    def targets(self) -> list[str]:
        return self._targets

    @property
    def train_time_range(self) -> tuple:
        return (self._train_time_min, self._train_time_max)

    def serialize(self) -> dict:
        return {
            "id_col":         self._id_col,
            "time_col":       self._time_col,
            "targets":        self._targets,
            "train_time_min": self._train_time_min if self._train_time_min is not None else None,
            "train_time_max": self._train_time_max if self._train_time_max is not None else None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "LMEModelSpec":
        return cls(
            data["id_col"],
            data["time_col"],
            data["targets"],
            data.get("train_time_min"),  # None if not yet set
            data.get("train_time_max"),
        )

class LMEModelPortObject(knext.PortObject):
    """
    data: {target_name: FittedTargetModel} — one fitted linear LME per
    target column selected in the Learner's dialog.
    """

    def __init__(self, spec: LMEModelSpec, fitted_models: dict[str, FittedTargetModel]):
        super().__init__(spec)
        self._fitted_models = fitted_models

    @property
    def fitted_models(self) -> dict[str, FittedTargetModel]:
        return self._fitted_models

    def serialize(self) -> bytes:
        payload = {
            target: {
                "target": fm.target,
                "id_col": fm.id_col,
                "time_col": fm.time_col,
                "fe_intercept": fm.fe_intercept,
                "fe_slope": fm.fe_slope,
                "cov_re": fm.cov_re,
                "scale": fm.scale,
                "training_ids": fm.training_ids,
                "converged": fm.converged,
            }
            for target, fm in self._fitted_models.items()
        }
        return json.dumps(payload).encode("utf-8")

    @classmethod
    def deserialize(cls, spec: LMEModelSpec, data: bytes) -> "LMEModelPortObject":
        from .lme_core import FittedTargetModel
        payload = json.loads(data.decode("utf-8"))
        fitted_models = {
            target: FittedTargetModel(**fields) for target, fields in payload.items()
        }
        return cls(spec, fitted_models)


if not hasattr(knext, "_lme_model_port_type_registered"):
    lme_model_port_type = knext.port_type(
        "LME Model", LMEModelPortObject, LMEModelSpec
    )
    knext._lme_model_port_type_registered = lme_model_port_type
else:
    lme_model_port_type = knext._lme_model_port_type_registered

