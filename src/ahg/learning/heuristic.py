"""A trained model as a search heuristic h(task, state) -> the regime's cost units (D21 §4; P9a2).

bind(instance) returns the callable best_first_search takes. Everything the
model needs per instance (the GNN's static graph) is built lazily inside the
first call, so it is inside the search's own timing, as rule 7 requires.
One state per call: no batching across successors, so the measured time is
what plain A* pays.

A checkpoint trained on the D28 encoder (config "features": "d28") binds only to a
d28encode.D28Instance and reads its inputs through the same d28encode functions training
uses; a D19 checkpoint binds only to a D19 instance (P7c, OQ-P7b-1).
"""

import torch

from ..dataset import mlp_features, static_graph, wl_colours
from ..heuristics import hmax
from . import d28encode
from .encode import fact_bits, gnn_inputs
from .models import WLLinear, goal_mask, output_transform, wl_transform


class Learned:
    """combine="max_hmax": search with max(h_max, h_θ) (D22 variant b); h_max is then inside the timing."""

    def __init__(self, model, cfg: dict, name: str, combine: str | None = None):
        assert combine in (None, "max_hmax")
        self.model, self.cfg, self.name, self.combine = model, cfg, name, combine

    def bind(self, inst):
        h = self._bind(inst)
        if self.combine == "max_hmax":
            return lambda task, s: max(hmax(task, s), h(task, s))
        return h

    def _bind(self, inst):
        kind, model = self.cfg["kind"], self.model
        s = float(model.label_scale)  # the model's unit, fitted on its training labels (P9a2)
        d28 = self.cfg.get("features") == d28encode.FEATURE_SET
        if d28 != isinstance(inst, d28encode.D28Instance):
            raise ValueError(f"{self.name}: a {'d28' if d28 else 'D19'}-feature checkpoint cannot bind to "
                             f"{type(inst).__name__} (features must match: d28 checkpoints on D28Instance only)")
        # The one place each model's per-state inputs come from; d28encode's are the functions training uses.
        colours = (lambda task, st: d28encode.wl_colours(inst, st)) if d28 else wl_colours
        features = (lambda task, st: d28encode.mlp_features(inst, st)) if d28 else mlp_features
        graph_fn = d28encode.static_graph if d28 else static_graph
        # Every branch maps the raw output through the same transform training uses (D28 LRN7), h(goal) = 0.
        if kind == "wl":
            assert isinstance(model, WLLinear)

            def h(task, st):
                raw = torch.tensor([model.raw_one(colours(task, st))], dtype=torch.float64)
                return s * float(wl_transform(raw, torch.tensor([task.is_goal(st)]))[0])
            return h
        if kind == "mlp":
            def h(task, st):
                with torch.no_grad():
                    raw = model(torch.tensor([features(task, st)], dtype=torch.float32))
                    return s * float(output_transform(raw, torch.tensor([task.is_goal(st)]))[0])
            return h
        graph = {}
        residual = self.cfg.get("target", "hstar") == "residual"

        def h(task, st):  # the static graph and (residual) h_max are computed here, inside the timing
            if not graph:
                graph.update(graph_fn(inst))
            with torch.no_grad():
                if residual:
                    x = gnn_inputs(task, [st], residual=True)
                    raw = model(x["fx"], graph, x["glob"])
                    return s * float(output_transform(raw, goal_mask(x["fx"]), x["hm"] / model.label_scale)[0])
                fx = fact_bits(task, [st])
                return s * float(output_transform(model(fx, graph), goal_mask(fx))[0])
        return h

    def predict_states(self, inst, states: list[int]) -> list[float]:
        """h_θ in the regime's cost units for many states of one instance (overestimation statistics; not timed)."""
        h = self.bind(inst)
        return [h(inst.task, s) for s in states]
