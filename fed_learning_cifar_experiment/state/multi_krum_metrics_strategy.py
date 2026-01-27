import json
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, Parameters
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fed_learning_cifar_experiment.utils.logger import (
    append_distributed_round,
    write_experiment_summary,
)


class SaveMultiKrumMetricsStrategy(fl.server.strategy.FedAvg):
    """
    Multi-Krum (true k-selection) strategy + your existing metrics/logging plumbing.

    Fixes applied vs initial draft:
    - Normalizes client update vectors before distance computation (recommended for stability)
    - Uses uniform averaging over selected clients (classic Multi-Krum)
    - Handles failures: falls back to FedAvg if n <= 2f + 2
    - Robust global-parameter reference:
        * Prefer the "parameters" argument passed to configure_fit for the round
        * Update stored global params AFTER aggregation, so next round is consistent
    """

    def __init__(
        self,
        simulation_id: str = "",
        num_clients: int = 0,
        num_rounds: int = 0,
        aggregation_method: str = "",
        backdoor_attack_mode: str = "",
        num_of_malicious_clients: int = 0,
        num_of_malicious_clients_per_round: int = 0,
        # Multi-Krum knobs
        num_byzantine: int = 0,              # f
        num_clients_to_select: int = 1,      # k
        # Numerical knobs
        normalize_updates: bool = True,
        eps: float = 1e-12,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # Experiment/meta
        self.simulation_id = simulation_id
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.aggregation_method = aggregation_method
        self.backdoor_attack_mode = backdoor_attack_mode

        # Histories
        self.history = {"round": [], "mta": [], "asr": []}
        self.central_mta_history: List[float] = []
        self.central_asr_history: List[float] = []
        self.final_centralized_mta: Optional[float] = None
        self.final_centralized_asr: Optional[float] = None

        # Attack sampling params
        self.num_of_malicious_clients = int(num_of_malicious_clients)
        self.num_of_malicious_clients_per_round = int(num_of_malicious_clients_per_round)

        # Multi-Krum params
        self.num_byzantine = int(num_byzantine)
        self.num_clients_to_select = int(num_clients_to_select)

        # Numerics
        self.normalize_updates = bool(normalize_updates)
        self.eps = float(eps)

        # Track the global parameters used to generate client updates for a round
        # (set in configure_fit; updated after aggregate_fit completes)
        self._global_parameters_for_round: Optional[Parameters] = None

    # ------------------------- Utility -------------------------
    @staticmethod
    def _flatten_ndarrays(nds: List[np.ndarray]) -> np.ndarray:
        # Use float64 for numerical stability in distance computations
        return np.concatenate([a.ravel() for a in nds]).astype(np.float64, copy=False)

    # ------------------------- Same malicious sampling/config injection -------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Save the global parameters that clients will train from this round
        self._global_parameters_for_round = parameters

        num_available = len(client_manager.all())
        sample_size, min_num = self.num_fit_clients(num_available)
        sampled_clients = list(client_manager.sample(sample_size, min_num))
        sampled_ids = [c.cid for c in sampled_clients]

        # Randomly pick malicious clients from the sampled list
        num_malicious = min(self.num_of_malicious_clients_per_round, len(sampled_ids))
        malicious_ids = random.sample(sampled_ids, num_malicious)

        fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
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

    # ------------------------- Multi-Krum core -------------------------
    def aggregate_fit(self, server_round: int, results, failures):
        """
        Multi-Krum:
        - Build normalized update vectors (theta_i - theta_global)
        - Compute Krum scores using m = n - f - 2 nearest neighbors
        - Select k smallest scores
        - Uniformly average selected client parameters
        - Fallback to FedAvg if not enough clients due to failures
        """
        if not results:
            return None, {}

        f = self.num_byzantine
        n = len(results)

        # Failure-aware guard: Krum/Multi-Krum requires n > 2f + 2
        if n <= 2 * f + 2:
            print(
                f"[Round {server_round}] [WARN] Multi-Krum needs n > 2f + 2, got n={n}, f={f}. "
                f"Falling back to FedAvg for this round."
            )
            aggregated = super().aggregate_fit(server_round, results, failures)
            # Update global params tracking if aggregation succeeded
            if aggregated and aggregated[0] is not None:
                self._global_parameters_for_round = aggregated[0]
            return aggregated

        if self._global_parameters_for_round is None:
            # Should not happen (configure_fit sets it), but guard against rare ordering issues
            print(
                f"[Round {server_round}] [WARN] Missing global parameters for round; "
                f"falling back to FedAvg."
            )
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated and aggregated[0] is not None:
                self._global_parameters_for_round = aggregated[0]
            return aggregated

        # Convert global parameters to vector
        global_nds = [np.asarray(a) for a in parameters_to_ndarrays(self._global_parameters_for_round)]
        global_vec = self._flatten_ndarrays(global_nds)

        # Gather client update vectors and client params
        client_params_nds: List[List[np.ndarray]] = []
        client_update_vecs: List[np.ndarray] = []
        client_cids: List[str] = []

        for client_proxy, fit_res in results:
            nds = [np.asarray(a) for a in parameters_to_ndarrays(fit_res.parameters)]
            vec = self._flatten_ndarrays(nds)
            update_vec = vec - global_vec

            if self.normalize_updates:
                norm = float(np.linalg.norm(update_vec))
                update_vec = update_vec / (norm + self.eps)

            client_params_nds.append(nds)
            client_update_vecs.append(update_vec)
            client_cids.append(getattr(client_proxy, "cid", "unknown"))

        # Multi-Krum scoring uses m = n - f - 2
        m = n - f - 2
        if m <= 0:
            # Should be prevented by the n > 2f + 2 check, but keep safe
            print(
                f"[Round {server_round}] [WARN] Invalid m={m} (n={n}, f={f}); falling back to FedAvg."
            )
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated and aggregated[0] is not None:
                self._global_parameters_for_round = aggregated[0]
            return aggregated

        # Pairwise squared Euclidean distances between update vectors
        dists = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            ui = client_update_vecs[i]
            for j in range(i + 1, n):
                diff = ui - client_update_vecs[j]
                dist = float(np.dot(diff, diff))
                dists[i, j] = dist
                dists[j, i] = dist

        # Krum scores: sum of m nearest (excluding self)
        scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            row = np.delete(dists[i], i)
            row.sort()
            scores[i] = float(np.sum(row[:m]))

        # Select k clients with smallest scores
        k = max(1, min(int(self.num_clients_to_select), n))
        selected_idx = np.argsort(scores)[:k].tolist()
        selected_cids = [client_cids[i] for i in selected_idx]

        print(
            f"[Round {server_round}] Multi-Krum selected k={k}/{n} (f={f}, m={m}). "
            f"Selected CIDs={selected_cids}"
        )

        # Uniform average of selected PARAMETERS (classic Multi-Krum)
        agg_nds = [np.zeros_like(arr) for arr in client_params_nds[selected_idx[0]]]
        for i in selected_idx:
            for layer_idx, layer in enumerate(client_params_nds[i]):
                agg_nds[layer_idx] += layer / float(k)

        new_parameters = ndarrays_to_parameters(agg_nds)

        # Update global params tracking to the new global params after aggregation
        self._global_parameters_for_round = new_parameters

        return new_parameters, {}

    # ------------------------- Same eval aggregation as your FedAvg class -------------------------
    def aggregate_evaluate(self, rnd, results, failures):
        metrics = super().aggregate_evaluate(rnd, results, failures)

        mta_vals = [res.metrics.get("mta", 0.0) for _, res in results]
        asr_vals = [res.metrics.get("asr", 0.0) for _, res in results]

        avg_mta = sum(mta_vals) / len(mta_vals) if mta_vals else 0.0
        avg_asr = sum(asr_vals) / len(asr_vals) if asr_vals else 0.0

        self.history["round"].append(rnd)
        self.history["mta"].append(avg_mta)
        self.history["asr"].append(avg_asr)

        print(f"[Round {rnd}] MTA={avg_mta:.4f}, ASR={avg_asr:.4f}")

        dist_loss = metrics[0] if metrics else None
        append_distributed_round(
            self.simulation_id,
            rnd,
            avg_mta,
            avg_asr,
            dist_loss,
            self.num_clients,
        )

        if rnd >= self.num_rounds:
            dist_mta = self.history.get("mta", [])
            dist_asr = self.history.get("asr", [])

            write_experiment_summary(
                simulation_id=self.simulation_id,
                meta={
                    "aggregation": str(self.aggregation_method),
                    "num_rounds": str(self.num_rounds),
                    # kept identical to your current behavior
                    "num_malicious_clients": str(self.num_clients),
                    "backdoor_attack_mode": str(self.backdoor_attack_mode),
                    "alpha": 0.9,
                },
                final_centralized_mta=self.final_centralized_mta or 0.0,
                final_centralized_asr=self.final_centralized_asr or 0.0,
                dist_mta_history=dist_mta,
                dist_asr_history=dist_asr,
                central_mta_history=self.central_mta_history,
                central_asr_history=self.central_asr_history,
                notes="",
            )

        return metrics

    def record_centralized_eval(self, rnd: int, loss: float, mta: float, asr: float) -> None:
        self.central_mta_history.append(mta)
        self.central_asr_history.append(asr)
        if rnd == self.num_rounds:
            self.final_centralized_mta = mta
            self.final_centralized_asr = asr