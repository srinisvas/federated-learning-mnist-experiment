from flwr.simulation import start_simulation
from fed_learning_mnist_experiment.client_app import app as client_app
from fed_learning_mnist_experiment.server_app import app as server_app
import flwr as fl

if __name__ == "__main__":
    start_simulation(
        client_fn=client_app,
        server_app=server_app,
        num_clients=10,   # number of clients
        config=fl.server.ServerConfig(num_rounds=5),
    )
