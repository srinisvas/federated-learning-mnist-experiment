import json
import random
from typing import Dict, List, Tuple, Optional

import flwr as fl
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays, ndarrays_to_parameters

import numpy as np

from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
)


class SaveKrumMetricsStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        simulation_id: str = "",
        num_clients: int = 0,
        num_rounds: int = 0,
        aggregation_method: str = "Krum",
        backdoor_attack_mode: str = "",
        num_of_malicious_clients: int = 0,
        num_of_malicious_clients_per_round: int = 0,
        num_byzantine: int = 0,  # f

        # FedAvg-compatible args
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters=None,

        # sampling knobs
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        fraction_evaluate: float = 1.0,
        min_evaluate_clients: int = 2,
    ):
        # metadata
        self.simulation_id = simulation_id
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.aggregation_method = aggregation_method
        self.backdoor_attack_mode = backdoor_attack_mode

        # histories
        self.history = {"round": [], "mta": [], "asr": []}
        self.central_mta_history = []
        self.central_asr_history = []
        self.final_centralized_mta = None
        self.final_centralized_asr = None

        # attack setup
        self.num_of_malicious_clients = num_of_malicious_clients
        self.num_of_malicious_clients_per_round = num_of_malicious_clients_per_round
        self.num_byzantine = int(num_byzantine)

        self._cid_to_partition: Dict[str, int] = {}

        # callbacks
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn

        self.accept_failures = bool(accept_failures)

        # parameters
        self.initial_parameters = initial_parameters
        self._latest_parameters = initial_parameters  # server params used as global reference

        # sampling
        self.fraction_fit = float(fraction_fit)
        self.min_fit_clients = int(min_fit_clients)
        self.min_available_clients = int(min_available_clients)

        self.fraction_evaluate = float(fraction_evaluate)
        self.min_evaluate_clients = int(min_evaluate_clients)

    # ---------------------------------------------------------------------
    # Strategy API (required)
    # ---------------------------------------------------------------------
    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def get_properties(self, ins):  # compatibility across Flower versions
        return {}

    def configure_get_properties(self, server_round: int, client_manager):  # compatibility
        return []

    def evaluate(self, server_round: int, parameters):
        """Server-side centralized evaluation (records metrics)."""
        if self.evaluate_fn is None:
            return None

        # Flower's evaluate_fn signature expects (server_round, parameters, config)
        config = {}
        res = self.evaluate_fn(server_round, parameters, config)

        if res is None:
            return None

        loss, metrics = res
        mta = metrics.get("mta", 0.0) if metrics else 0.0
        asr = metrics.get("asr", 0.0) if metrics else 0.0

        self.record_centralized_eval(server_round, loss, mta, asr)
        return loss, metrics

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Sample clients for training + mark random subset as malicious."""
        # store the true server global parameters for this round (important for deltas)
        self._latest_parameters = parameters

        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)
        sampled_clients = list(client_manager.sample(sample_size, min_num))
        sampled_ids = [c.cid for c in sampled_clients]

        num_malicious = min(self.num_of_malicious_clients_per_round, len(sampled_ids))
        malicious_ids = random.sample(sampled_ids, num_malicious)

        fit_ins_list = []
        for client in sampled_clients:
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            config.update(
                {
                    "current-round": server_round,
                    "sampled_client_ids": json.dumps(sampled_ids),
                    "malicious_client_ids": json.dumps(malicious_ids),
                    "is_malicious": str(client.cid in malicious_ids),
                }
            )
            fit_ins_list.append((client, FitIns(parameters, config)))

        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        """Single-Krum aggregation with safe fallback (no mean-based selection)."""
        if not results:
            return None, {}

        if failures and not self.accept_failures:
            return None, {}

        if self._latest_parameters is None:
            raise RuntimeError("Missing _latest_parameters. configure_fit must run before aggregate_fit.")

        # global weights for this round
        w_global = parameters_to_ndarrays(self._latest_parameters)

        # build updates (deltas) and keep full client params
        updates: List[List[np.ndarray]] = []
        client_params: List[List[np.ndarray]] = []

        for _, fit_res in results:
            w_client = parameters_to_ndarrays(fit_res.parameters)
            delta = [wc - wg for wc, wg in zip(w_client, w_global)]
            updates.append(delta)
            client_params.append(w_client)

        flat_updates = [self._flatten_update(u) for u in updates]

        chosen_idx, f_eff = self._krum_select_index(flat_updates, f=self.num_byzantine)

        if chosen_idx is not None:
            # single-Krum: take chosen client's weights
            new_params = ndarrays_to_parameters(client_params[chosen_idx])
            self._latest_parameters = new_params

            print(
                f"[Round {server_round}] Krum selected idx={chosen_idx} "
                f"(n={len(results)}, f={self.num_byzantine}->{f_eff}, failures={len(failures) if failures else 0})"
            )
            return new_params, {"krum_selected_client_index": chosen_idx, "f_eff": f_eff}

        # Safe fallback: coordinate-wise median of updates, applied to global
        flat_agg = self._coord_median(flat_updates)
        agg_delta = self._unflatten_update(flat_agg, updates[0])
        w_new = [wg + du for wg, du in zip(w_global, agg_delta)]
        new_params = ndarrays_to_parameters(w_new)
        self._latest_parameters = new_params

        print(
            f"[Round {server_round}] Krum unsafe for requested f={self.num_byzantine} "
            f"(n={len(results)}). Used coord-median fallback."
        )
        return new_params, {"krum_selected_client_index": -1, "fallback": "coord_median"}

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Client-side evaluation scheduling (required by your Flower version)."""
        num_available = len(client_manager.all())
        sample_size, min_num = self.num_evaluation_clients(num_available)
        if sample_size <= 0:
            return []

        sampled_clients = list(client_manager.sample(sample_size, min_num))
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        evaluate_ins = EvaluateIns(parameters, config)
        return [(client, evaluate_ins) for client in sampled_clients]

    def aggregate_evaluate(self, rnd, results, failures):
        """Distributed eval aggregation + logging + final summary writing."""
        if not results:
            return None

        mta_vals = [res.metrics.get("mta", 0.0) for _, res in results]
        asr_vals = [res.metrics.get("asr", 0.0) for _, res in results]

        avg_mta = sum(mta_vals) / len(mta_vals) if mta_vals else 0.0
        avg_asr = sum(asr_vals) / len(asr_vals) if asr_vals else 0.0

        self.history["round"].append(rnd)
        self.history["mta"].append(avg_mta)
        self.history["asr"].append(avg_asr)

        print(f"[Round {rnd}] MTA={avg_mta:.4f}, ASR={avg_asr:.4f}")

        # weighted average of client losses (if provided)
        loss_vals = [(res.num_examples, res.loss) for _, res in results]
        total = sum(n for n, _ in loss_vals)
        dist_loss = sum(n * loss for n, loss in loss_vals) / total if total > 0 else None

        append_distributed_round(
            self.simulation_id,
            rnd,
            avg_mta,
            avg_asr,
            dist_loss,
            self.num_clients,
        )

        if rnd >= self.num_rounds:
            write_experiment_summary(
                simulation_id=self.simulation_id,
                meta={
                    "aggregation": str(self.aggregation_method),
                    "num_rounds": str(self.num_rounds),
                    "num_malicious_clients": str(self.num_of_malicious_clients),
                    "backdoor_attack_mode": str(self.backdoor_attack_mode),
                    "alpha": 0.9,
                    "krum_num_byzantine": str(self.num_byzantine),
                },
                final_centralized_mta=self.final_centralized_mta or 0.0,
                final_centralized_asr=self.final_centralized_asr or 0.0,
                dist_mta_history=self.history.get("mta", []),
                dist_asr_history=self.history.get("asr", []),
                central_mta_history=self.central_mta_history,
                central_asr_history=self.central_asr_history,
                notes="",
            )

        return dist_loss, {"mta": avg_mta, "asr": avg_asr}

    # ---------------------------------------------------------------------
    # Tracking centralized metrics
    # ---------------------------------------------------------------------
    def record_centralized_eval(self, rnd, loss, mta, asr):
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr

    # ---------------------------------------------------------------------
    # Client sampling helpers
    # ---------------------------------------------------------------------
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        sample_size = int(num_available_clients * self.fraction_fit)
        sample_size = max(sample_size, self.min_fit_clients)
        return sample_size, self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        sample_size = int(num_available_clients * self.fraction_evaluate)
        sample_size = max(sample_size, self.min_evaluate_clients)
        return sample_size, self.min_available_clients

    # ---------------------------------------------------------------------
    # Krum helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _flatten_update(update: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([u.reshape(-1).astype(np.float32, copy=False) for u in update])

    @staticmethod
    def _unflatten_update(flat: np.ndarray, like_update: List[np.ndarray]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        idx = 0
        for arr in like_update:
            size = arr.size
            out.append(flat[idx:idx + size].reshape(arr.shape).astype(arr.dtype, copy=False))
            idx += size
        return out

    @staticmethod
    def _coord_median(flat_updates: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack(flat_updates, axis=0)
        return np.median(stacked, axis=0).astype(np.float32, copy=False)

    def _krum_select_index(self, flat_updates: List[np.ndarray], f: int) -> Tuple[Optional[int], int]:
        """
        Returns (chosen_index, f_eff).
        If Krum cannot be applied safely for the requested f, we reduce f to max allowed.
        If f_eff becomes 0, we return (None, f_eff) and let aggregate_fit use robust fallback.
        """
        n = len(flat_updates)
        if n == 0:
            return None, 0
        if n == 1:
            return 0, 0

        f = int(f)
        # Krum needs n >= 2f + 3
        max_safe_f = (n - 3) // 2
        if max_safe_f < 0:
            return None, 0

        f_eff = min(f, max_safe_f)

        # If f_eff == 0, Krum gives no Byzantine tolerance. Use robust fallback instead.
        if f_eff == 0:
            return None, f_eff

        # Pairwise squared L2 distances
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                diff = flat_updates[i] - flat_updates[j]
                dij = float(np.dot(diff, diff))
                dist[i, j] = dij
                dist[j, i] = dij

        # score: sum of closest (n - f_eff - 2) distances
        m = n - f_eff - 2
        scores = []
        for i in range(n):
            dists = np.sort(dist[i])[1:]  # exclude self
            scores.append(np.sum(dists[:m]))

        return int(np.argmin(scores)), f_eff