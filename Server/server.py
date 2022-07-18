import flwr as fl
from typing import List,Tuple
import numpy as np
import os
from flwr.server.strategy.aggregate import aggregate

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):

    global list_of_clients
    list_of_clients = [] 

    
    
    def configure_fit(
        self, rnd, parameters, client_manager):
        """Configure the next round of training."""
        #global list_of_clients

        config = {
            "selected_clients" : ' '.join(list_of_clients[:int(os.environ['CLIENTS_TO_SELECT'])])
        }

        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = fl.common.FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):

        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = []
        for _, fit_res in results:
            
            client_id = str(fit_res.metrics['client_id'])

            if int(os.environ['POC']) == 0 or int(rnd) <= 1:
                weights_results.append((fl.common.parameters_to_weights(fit_res.parameters), fit_res.num_examples))

            else:
                if client_id in list_of_clients[:int(os.environ['CLIENTS_TO_SELECT'])]:
                    weights_results.append((fl.common.parameters_to_weights(fit_res.parameters), fit_res.num_examples))
        
        
        parameters_aggregated = fl.common.weights_to_parameters(aggregate(weights_results))
        metrics_aggregated    = {}

        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        global list_of_clients

        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        list_of_clients = []

        for response in results:
            #client_ip      = response[0].cid
            client_id       = response[1].metrics['client_id']
            client_accuracy = response[1].metrics['accuracy']

            list_of_clients.append((client_id, client_accuracy))

        list_of_clients.sort(key=lambda x: x[1])
        list_of_clients = [str(client[0]) for client in list_of_clients]

        print(f'List of Clients: {list_of_clients}')

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples   = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


if __name__ == "__main__":

    server_fraction_fit          = float(os.environ['FRACTION_FIT'])
    server_min_fit_clients       = int(os.environ['MIN_FIT_CLIENTS'])
    server_min_available_clients = int(os.environ['MIN_AVAILABLE_CLIENTS'])
    server_min_eval_clients      = int(os.environ['MIN_EVAL_CLIENTS'])

    server_num_rounds = int(os.environ['NUM_ROUNDS'])

    server_ip = os.environ['SERVER_IP']

    strategy = AggregateCustomMetricStrategy(
        fraction_fit          = server_fraction_fit,  # Sample % of available clients for the next round (0.1 = 10%)
        min_fit_clients       = server_min_fit_clients,  # Minimum number of clients to be sampled for the next round
        min_available_clients = server_min_available_clients,  # Minimum number of clients that need to be connected to the server before a training round can start
        min_eval_clients      = server_min_eval_clients,
    )

    fl.server.start_server(
        server_ip, 
        config   = {"num_rounds": server_num_rounds}, 
        strategy = strategy, 
        )