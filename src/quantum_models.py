"""
Quantum Machine Learning models for fraud detection.

Models:
1. QSVM: Quantum Kernel SVM using PennyLane quantum kernel + sklearn SVC
2. VQC: Variational Quantum Classifier with data re-uploading
3. Hybrid QNN: Classical layers -> Quantum layer -> Classical output

All models use PennyLane's default.qubit simulator and are designed to work
with 6 qubits (TOP_N_FEATURES from config).
"""
import time
import numpy as np
from pennylane import numpy as pnp
import pennylane as qml
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    N_QUBITS, N_LAYERS_QSVM, N_LAYERS_VQC, N_LAYERS_HYBRID,
    VQC_LEARNING_RATE, VQC_EPOCHS, VQC_BATCH_SIZE,
    HYBRID_LEARNING_RATE, HYBRID_EPOCHS, RANDOM_SEED
)


# ============================================================================
# MODEL 1: QUANTUM KERNEL SVM (QSVM)
# ============================================================================

class QuantumKernelSVM(BaseEstimator, ClassifierMixin):
    """
    Quantum Support Vector Machine using a quantum kernel.

    The quantum kernel computes similarity between data points by:
    1. Encoding x1 into a quantum state via a feature map
    2. Applying the adjoint of the feature map for x2
    3. Measuring the probability of returning to |0...0⟩

    k(x1, x2) = |⟨φ(x2)|φ(x1)⟩|²

    This kernel captures correlations in a high-dimensional Hilbert space
    that classical kernels cannot efficiently access.
    """

    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS_QSVM,
                 C=1.0, feature_map="ZZ"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.C = C
        self.feature_map = feature_map
        self.svc_ = None
        self.train_time_ = 0

        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Build quantum kernel circuit
        @qml.qnode(self.dev)
        def _kernel_circuit(x1, x2):
            self._apply_feature_map(x1)
            qml.adjoint(self._apply_feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))

        self._kernel_circuit = _kernel_circuit

    def _apply_feature_map(self, x):
        """
        Apply quantum feature map to encode classical data.

        ZZ Feature Map: Applies Hadamard gates, then RZ rotations with data,
        then ZZ entangling interactions between neighboring qubits.
        This creates entanglement that depends on the input data,
        enabling the kernel to capture feature correlations.
        """
        for layer in range(self.n_layers):
            # Layer of Hadamard gates
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Data encoding with RZ rotations
            for i in range(self.n_qubits):
                qml.RZ(x[i], wires=i)

            # ZZ entanglement (data-dependent)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x[i] * x[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

            # Add circular entanglement for richer kernel
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
                qml.RZ(x[-1] * x[0], wires=0)
                qml.CNOT(wires=[self.n_qubits - 1, 0])

    def _compute_kernel_entry(self, x1, x2):
        """Compute single kernel matrix entry k(x1, x2)."""
        return float(self._kernel_circuit(x1, x2)[0])

    def _compute_kernel_matrix(self, X1, X2):
        """Compute the full kernel (Gram) matrix between X1 and X2."""
        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._compute_kernel_entry(X1[i], X2[j])

        return K

    def fit(self, X, y):
        """
        Fit the QSVM model.

        1. Compute the quantum kernel matrix for training data
        2. Fit a classical SVM using the precomputed kernel
        """
        start_time = time.time()
        self.X_train_ = X.copy()

        # Compute training kernel matrix
        print(f"   Computing quantum kernel matrix ({len(X)}x{len(X)})...")
        K_train = self._compute_kernel_matrix(X, X)

        # Fit SVM with precomputed kernel
        self.svc_ = SVC(kernel="precomputed", C=self.C, random_state=RANDOM_SEED)
        self.svc_.fit(K_train, y)

        self.train_time_ = time.time() - start_time
        print(f"   QSVM training complete in {self.train_time_:.1f}s")
        return self

    def predict(self, X):
        """Predict class labels for X."""
        K_test = self._compute_kernel_matrix(X, self.X_train_)
        return self.svc_.predict(K_test)

    def predict_proba(self, X):
        """
        Approximate predict_proba using SVM decision function.
        Convert decision values to pseudo-probabilities via sigmoid.
        """
        K_test = self._compute_kernel_matrix(X, self.X_train_)
        decision = self.svc_.decision_function(K_test)
        # Sigmoid to convert to pseudo-probability
        prob_pos = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - prob_pos, prob_pos])


# ============================================================================
# MODEL 2: VARIATIONAL QUANTUM CLASSIFIER (VQC)
# ============================================================================

class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """
    Variational Quantum Classifier with data re-uploading.

    Architecture:
    - Data encoding layer (AngleEmbedding)
    - Parameterized variational layers (StronglyEntanglingLayers)
    - Repeat encoding + variational for each layer (data re-uploading)
    - Measurement on first qubit

    Data re-uploading allows the circuit to create more complex
    decision boundaries than single-encoding approaches.
    """

    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS_VQC,
                 learning_rate=VQC_LEARNING_RATE, epochs=VQC_EPOCHS,
                 batch_size=VQC_BATCH_SIZE):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights_ = None
        self.bias_ = None
        self.train_time_ = 0
        self.loss_history_ = []

        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Build quantum circuit
        @qml.qnode(self.dev, interface="autograd")
        def _circuit(weights, bias, x):
            # Data re-uploading: alternate encoding and variational layers
            for layer in range(self.n_layers):
                # Data encoding
                qml.AngleEmbedding(x, wires=range(self.n_qubits))

                # Variational layer
                qml.StronglyEntanglingLayers(
                    weights[layer:layer+1], wires=range(self.n_qubits)
                )

            # Measurement: expectation of PauliZ on first qubit
            return qml.expval(qml.PauliZ(0))

        self._circuit = _circuit

    def _cost_single(self, weights, bias, x, y):
        """Binary cross-entropy loss for a single sample."""
        # Circuit output in [-1, 1], shift to [0, 1]
        prediction = (self._circuit(weights, bias, x) + 1) / 2
        # Clip to avoid log(0) — use pnp for autograd compatibility
        prediction = pnp.clip(prediction, 1e-7, 1 - 1e-7)
        return -y * pnp.log(prediction) - (1 - y) * pnp.log(1 - prediction)

    def _cost_batch(self, weights, bias, X, y):
        """Average cost over a batch."""
        costs = [self._cost_single(weights, bias, X[i], y[i])
                 for i in range(len(X))]
        return pnp.mean(pnp.array(costs))

    def fit(self, X, y):
        """
        Train the VQC using gradient descent.

        Uses PennyLane's automatic differentiation for quantum gradients.
        """
        start_time = time.time()

        # Initialize weights
        rng = np.random.RandomState(RANDOM_SEED)
        weight_shape = (self.n_layers, 1, self.n_qubits, 3)
        self.weights_ = qml.numpy.array(
            rng.uniform(-np.pi, np.pi, weight_shape), requires_grad=True
        )
        self.bias_ = qml.numpy.array(0.0, requires_grad=True)

        # Optimizer
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)

        self.loss_history_ = []
        n_samples = len(X)

        for epoch in range(self.epochs):
            # Shuffle data
            perm = rng.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            epoch_losses = []

            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Gradient step
                (self.weights_, self.bias_), cost = opt.step_and_cost(
                    lambda w, b: self._cost_batch(w, b, X_batch, y_batch),
                    self.weights_, self.bias_
                )
                epoch_losses.append(float(cost))

            avg_loss = np.mean(epoch_losses)
            self.loss_history_.append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = accuracy_score(y, self.predict(X))
                print(f"   VQC Epoch {epoch+1}/{self.epochs} | "
                      f"Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

        self.train_time_ = time.time() - start_time
        print(f"   VQC training complete in {self.train_time_:.1f}s")
        return self

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities."""
        probs_positive = []
        for x in X:
            raw = float(self._circuit(self.weights_, self.bias_, x))
            prob = (raw + 1) / 2  # Map [-1,1] to [0,1]
            prob = np.clip(prob, 0, 1)
            probs_positive.append(prob)
        probs_positive = np.array(probs_positive)
        return np.column_stack([1 - probs_positive, probs_positive])


# ============================================================================
# MODEL 3: HYBRID QUANTUM-CLASSICAL NEURAL NETWORK
# ============================================================================

class HybridQuantumNN(BaseEstimator, ClassifierMixin):
    """
    Hybrid Quantum-Classical Neural Network.

    Architecture:
    1. Classical linear layer: n_features -> n_qubits (dimensionality adapter)
    2. Quantum variational circuit: n_qubits -> 1 expectation value
    3. Classical output layer: quantum output + bias -> prediction

    This combines classical preprocessing with quantum feature processing.
    """

    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS_HYBRID,
                 learning_rate=HYBRID_LEARNING_RATE, epochs=HYBRID_EPOCHS):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.params_ = None
        self.train_time_ = 0
        self.loss_history_ = []

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def _quantum_layer(inputs, weights):
            # Encode pre-processed inputs
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        self._quantum_layer = _quantum_layer

    def _forward(self, params, x):
        """Forward pass through the hybrid network."""
        W_in, W_q, w_out, b_out = params

        # Classical pre-processing layer (linear) — use pnp for autograd
        x_processed = pnp.tanh(x @ W_in)

        # Clip to [-pi, pi] for quantum encoding
        x_processed = pnp.clip(x_processed, -pnp.pi, pnp.pi)

        # Quantum layer
        q_out = self._quantum_layer(x_processed, W_q)

        # Classical output — keep computation graph intact (no float())
        output = q_out * w_out + b_out
        return output

    def _sigmoid(self, z):
        """Numerically stable sigmoid — autograd compatible."""
        z_clipped = pnp.clip(z, -500, 500)
        return 1 / (1 + pnp.exp(-z_clipped))

    def _cost(self, params, X, y):
        """Binary cross-entropy loss — autograd compatible."""
        total_loss = pnp.array(0.0)
        for i in range(len(X)):
            logit = self._forward(params, X[i])
            prob = self._sigmoid(logit)
            prob = pnp.clip(prob, 1e-7, 1 - 1e-7)
            total_loss = total_loss + (-y[i] * pnp.log(prob) - (1 - y[i]) * pnp.log(1 - prob))
        return total_loss / len(X)

    def fit(self, X, y):
        """Train the hybrid quantum-classical neural network."""
        start_time = time.time()
        rng = np.random.RandomState(RANDOM_SEED)
        n_features = X.shape[1]

        # Initialize parameters
        W_in = qml.numpy.array(
            rng.uniform(-0.5, 0.5, (n_features, self.n_qubits)),
            requires_grad=True
        )
        W_q = qml.numpy.array(
            rng.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits, 3)),
            requires_grad=True
        )
        w_out = qml.numpy.array(1.0, requires_grad=True)
        b_out = qml.numpy.array(0.0, requires_grad=True)

        self.params_ = [W_in, W_q, w_out, b_out]

        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        self.loss_history_ = []

        for epoch in range(self.epochs):
            self.params_, cost = opt.step_and_cost(
                lambda *p: self._cost(list(p), X, y),
                *self.params_
            )
            self.params_ = list(self.params_)
            self.loss_history_.append(float(cost))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = accuracy_score(y, self.predict(X))
                print(f"   Hybrid QNN Epoch {epoch+1}/{self.epochs} | "
                      f"Loss: {float(cost):.4f} | Train Acc: {acc:.4f}")

        self.train_time_ = time.time() - start_time
        print(f"   Hybrid QNN training complete in {self.train_time_:.1f}s")
        return self

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities."""
        probs_pos = []
        for x in X:
            logit = self._forward(self.params_, x)
            prob = float(self._sigmoid(logit))
            probs_pos.append(prob)
        probs_pos = np.array(probs_pos)
        return np.column_stack([1 - probs_pos, probs_pos])


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_quantum_models():
    """
    Return a dictionary of quantum models ready for training.

    Returns:
        dict: {model_name: model_instance}
    """
    return {
        "QSVM": QuantumKernelSVM(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS_QSVM,
            C=1.0,
        ),
        "VQC": VariationalQuantumClassifier(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS_VQC,
            learning_rate=VQC_LEARNING_RATE,
            epochs=VQC_EPOCHS,
            batch_size=VQC_BATCH_SIZE,
        ),
        "Hybrid_QNN": HybridQuantumNN(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS_HYBRID,
            learning_rate=HYBRID_LEARNING_RATE,
            epochs=HYBRID_EPOCHS,
        ),
    }
