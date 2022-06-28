import flwr as fl
from typing import List,Tuple
import numpy as np
import os

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)

if __name__ == "__main__":

    server_fraction_fit          = float(os.environ['FRACTION_FIT'])
    server_min_fit_clients       = int(os.environ['MIN_FIT_CLIENTS'])
    server_min_available_clients = int(os.environ['MIN_AVAILABLE_CLIENTS'])

    server_num_rounds = int(os.environ['NUM_ROUNDS'])

    server_ip = os.environ['SERVER_IP']

    strategy = AggregateCustomMetricStrategy(
        fraction_fit=server_fraction_fit,  # Sample % of available clients for the next round (0.1 = 10%)
        min_fit_clients=server_min_fit_clients,  # Minimum number of clients to be sampled for the next round
        min_available_clients=server_min_available_clients,  # Minimum number of clients that need to be connected to the server before a training round can start
    )

    fl.server.start_server(server_ip, config={"num_rounds": server_num_rounds}, strategy=strategy)