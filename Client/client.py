from collections import OrderedDict
from typing import Dict, List, Tuple
import os

import numpy as np
import torch

import motionsense
import compression_utils
import flwr as fl

# DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE     = 'cpu'
device     = 'cpu'
Print_bits = True

class MotionSenseClient(fl.client.NumPyClient):
    """Flower client implementing motionsense-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: motionsense.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
        compression_method : str
    ) -> None:
        self.model              = model
        self.trainloader        = trainloader
        self.testloader         = testloader
        self.num_examples       = num_examples
        self.compression_method = compression_method

    def get_parameters(self) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        motionsense.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        parametros = self.get_parameters()
        
        #selects de compression method to use
        if self.compression_method == 'STC':
            T = compression_utils.STC(self.get_parameters(), 0.25, 0.8)
            self.set_parameters(T)

        elif self.compression_method == 'DGC':
            T = compression_utils.DGC(self.get_parameters(), 0.25, 0.8)
            self.set_parameters(T)

        elif self.compression_method == 'SSGD':
            T = compression_utils.SSGD(self.get_parameters(), 0.25, 0.8)
            self.set_parameters(T)

        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = motionsense.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}



def main() -> None:
    """Load data, start motionsenseClient."""

    # Load model and data
    model = motionsense.Net()
    model.to(DEVICE)
    trainloader, testloader, num_examples = motionsense.load_data('MotionSense.csv')

    # Start client
    server_ip          = os.environ['SERVER_IP']
    compression_method = os.environ['COMPRESSION_METHOD']
    client             = MotionSenseClient(model, trainloader, testloader, num_examples, compression_method)
    fl.client.start_numpy_client(server_ip, client)


if __name__ == "__main__":
    main()