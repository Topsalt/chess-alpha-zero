"""
Pipe-based prediction API for the KaryotypeModel.

Analogous to api_chess.py: a background thread listens on a set of multiprocessing
Pipes, batches incoming state tensors, runs a forward pass through the Keras
policy+value network, and sends back the results.
"""
from multiprocessing import connection, Pipe
from threading import Thread

import numpy as np


class KaryotypeModelAPI:
    """
    Listens on a set of Pipe connections for encoded karyotype states and
    returns (policy, value) predictions from the shared KaryotypeModel.

    Attributes
    ----------
    agent_model : KaryotypeModel
        The shared model used for prediction.
    pipes : list[Connection]
        The server-side ends of all created pipes.
    """

    def __init__(self, agent_model):
        self.agent_model = agent_model
        self.pipes = []

    def start(self):
        """Start the background prediction thread (daemon)."""
        worker = Thread(target=self._predict_batch_worker,
                        name="karyotype_prediction_worker")
        worker.daemon = True
        worker.start()

    def create_pipe(self):
        """
        Create a new bidirectional pipe.

        The caller receives the *client* end; this API keeps the *server* end.

        Returns
        -------
        Connection
            Client end of the pipe to pass to the MCTS player.
        """
        server_end, client_end = Pipe()
        self.pipes.append(server_end)
        return client_end

    def _predict_batch_worker(self):
        """
        Background thread: poll all pipes, batch any available state tensors,
        run a single model.predict_on_batch() call, and send results back.

        Each state tensor has shape ``(STATE_DIM,)`` = ``(1198,)``
        (see KaryotypeEnv.encode_state).  The batch is stacked to
        ``(batch, STATE_DIM)`` before prediction.
        """
        while True:
            ready = connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue

            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)

            # data[i] shape: (N_CHROMOSOMES, input_dim)
            batch = np.asarray(data, dtype=np.float32)  # (B, N, D)
            policy_batch, value_batch = self.agent_model.model.predict_on_batch(batch)

            for pipe, policy, value in zip(result_pipes, policy_batch, value_batch):
                pipe.send((policy, float(value)))
