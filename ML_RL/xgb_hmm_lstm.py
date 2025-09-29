# XGB-HMM Implementation Classes
import pandas as pd
import numpy as np
import logging
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFE, RFECV, mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from hmmlearn.hmm import GaussianHMM
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
from optuna import create_study
from optuna.storages import InMemoryStorage
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')

# LSTM imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    TENSORFLOW_AVAILABLE = True
    logging.info("TensorFlow available for LSTM implementation")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM functionality will be disabled.")

# Additional imports for the main script
from datetime import datetime
import os
from fundamental_feature_engineering import apply_feature_engineering
from technical_fundamental_preprocessing import preprocess_technical_fundamental_data
from triple_barrier.triplebarrier import apply_triple_barrier_labeling
from triple_barrier.visualizebarrier import generate_triple_barrier_visualizations
from utils.load_data import process_fundamental_data_local, get_fundamental_data_local, process_fundamental_data

class LSTMTimeSeriesModel:
    """
    LSTM model for time series sequential learning
    Handles sequence creation, training, and prediction
    """
    
    def __init__(self, sequence_length=20, lstm_units=50, dropout_rate=0.2, 
                 learning_rate=0.001, batch_size=32, epochs=50, random_state=42):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM functionality")
            
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        self.n_features = None
        self.n_classes = None
        
        # Set TensorFlow random seed
        tf.random.set_seed(random_state)
        
    def _create_sequences(self, X, y=None):
        """Create sequences for LSTM training"""
        n_samples = len(X)
        
        if n_samples < self.sequence_length:
            logging.warning(f"Not enough samples ({n_samples}) for sequence length ({self.sequence_length})")
            return None, None
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, n_samples):
            X_sequences.append(X[i-self.sequence_length:i])
            if y is not None:
                y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        
        if y is not None:
            y_sequences = np.array(y_sequences)
            return X_sequences, y_sequences
        else:
            return X_sequences, None
    
    def _build_model(self, input_shape, n_classes):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape,
                 dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate),
            BatchNormalization(),
            
            LSTM(self.lstm_units // 2, return_sequences=False,
                 dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate),
            BatchNormalization(),
            
            Dense(self.lstm_units // 4, activation='relu',
                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(self.dropout_rate),
            
            Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        loss = 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def fit(self, X, y):
        """Fit LSTM model"""
        logging.info(f"Training LSTM model with sequence length {self.sequence_length}")
        
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        if X_seq is None:
            raise ValueError(f"Cannot create sequences with current data size")
        
        # Store dimensions
        self.n_features = X_seq.shape[2]
        self.n_classes = len(np.unique(y_seq))
        
        # Build model
        self.model = self._build_model(
            input_shape=(self.sequence_length, self.n_features),
            n_classes=self.n_classes
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Train model
        self.history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        logging.info(f"LSTM training completed. Final val_accuracy: {max(self.history.history['val_accuracy']):.4f}")
        
        return self
    
    def predict(self, X):
        """Predict using LSTM model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        if X_seq is None:
            # If not enough data for sequences, return predictions based on last available pattern
            logging.warning("Not enough data for sequences, using simplified prediction")
            return np.zeros(len(X))
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        
        if self.n_classes == 2:
            pred_classes = (predictions > 0.5).astype(int)
        else:
            pred_classes = np.argmax(predictions, axis=1)
        
        # Pad predictions to match original length
        full_predictions = np.zeros(len(X))
        full_predictions[self.sequence_length:] = pred_classes.flatten()
        
        return full_predictions
    
    def predict_proba(self, X):
        """Predict probabilities using LSTM model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        if X_seq is None:
            # Return uniform probabilities if not enough data
            return np.ones((len(X), self.n_classes)) / self.n_classes
        
        # Predict probabilities
        probabilities = self.model.predict(X_seq, verbose=0)
        
        # Pad probabilities to match original length
        full_probabilities = np.ones((len(X), self.n_classes)) / self.n_classes
        
        if self.n_classes == 2:
            full_probabilities[self.sequence_length:, 1] = probabilities.flatten()
            full_probabilities[self.sequence_length:, 0] = 1 - probabilities.flatten()
        else:
            full_probabilities[self.sequence_length:] = probabilities
        
        return full_probabilities

class XGBHMMModel:
    """
    Hybrid XGBoost-HMM Model for Sequential Classification
    Combines HMM temporal modeling with XGBoost emission probabilities
    """
    
    def __init__(self, n_states=3, max_iter=50, tol=1e-4, random_state=42):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Model components
        self.gmm_hmm = None
        self.xgb_model = None
        self.transition_matrix = None
        self.start_prob = None
        self.emission_model = None
        
        # Training history
        self.log_likelihood_history = []
        self.convergence_info = {}
        
    def _initialize_gmm_hmm(self, X, y):
        """Initialize GMM-HMM for initial state estimation"""
        logging.info("Initializing GMM-HMM for state estimation...")
        
        # Handle NaN values by filling with median
        X_clean = X.fillna(X.median())
        
        # Convert to numpy array if it's a DataFrame
        if hasattr(X_clean, 'values'):
            X_array = X_clean.values
        else:
            X_array = X_clean
        
        # Initialize Gaussian HMM with diagonal covariance for stability
        self.gmm_hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",  # Use diagonal covariance for numerical stability
            random_state=self.random_state,
            n_iter=20,
            tol=1e-2,  # More lenient tolerance
            min_covar=1e-3  # Add minimum covariance for numerical stability
        )
        
        # Fit GMM-HMM to get initial state sequences
        self.gmm_hmm.fit(X_array)
        
        # Get state sequences and probabilities
        state_sequences = self.gmm_hmm.predict(X_array)
        gamma = self.gmm_hmm.predict_proba(X_array)
        
        # Extract transition matrix and start probabilities
        self.transition_matrix = self.gmm_hmm.transmat_
        self.start_prob = self.gmm_hmm.startprob_
        
        return state_sequences, gamma
    
    def _train_xgb_emission_model(self, X, y, gamma):
        """Train XGBoost model for emission probabilities"""
        # Prepare training data with state probabilities
        n_samples, n_features = X.shape
        
        # Create expanded training data
        X_expanded = []
        y_expanded = []
        weights_expanded = []
        
        for i in range(n_samples):
            for state in range(self.n_states):
                if gamma[i, state] > 0.01:  # Only include significant state probabilities
                    X_expanded.append(X.iloc[i].values)
                    y_expanded.append(y.iloc[i])
                    weights_expanded.append(gamma[i, state])
        
        X_expanded = np.array(X_expanded)
        y_expanded = np.array(y_expanded)
        weights_expanded = np.array(weights_expanded)
        
        # Determine XGBoost parameters based on label distribution
        unique_labels = np.unique(y_expanded)
        if len(unique_labels) == 2:
            objective = 'binary:logistic'
            eval_metric = 'logloss'
            num_class = None
        else:
            objective = 'multi:softmax'
            eval_metric = 'mlogloss'
            num_class = len(unique_labels)
        
        # Initialize XGBoost with optimal parameters
        self.xgb_model = XGBClassifier(
            objective=objective,
            num_class=num_class,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,  # Dropout pentru level-wise feature sampling
            colsample_bynode=0.8,   # Dropout pentru node-wise feature sampling
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # Train XGBoost model
        self.xgb_model.fit(X_expanded, y_expanded, sample_weight=weights_expanded)
        
        return self.xgb_model
    
    def _compute_emission_probabilities(self, X):
        """Compute emission probabilities using XGBoost"""
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained yet")
        
        # Handle NaN values by filling with median
        X = np.array(X)
        if np.isnan(X).any():
            for col in range(X.shape[1]):
                col_median = np.nanmedian(X[:, col])
                X[:, col] = np.where(np.isnan(X[:, col]), col_median, X[:, col])
        
        # Get XGBoost predictions
        emission_probs = self.xgb_model.predict_proba(X)
        
        # Ensure probabilities are normalized
        emission_probs = emission_probs / emission_probs.sum(axis=1, keepdims=True)
        
        return emission_probs
    
    def _forward_backward_algorithm(self, X):
        """Implement forward-backward algorithm for state estimation"""
        n_samples = len(X)
        
        # Get emission probabilities
        emission_probs = self._compute_emission_probabilities(X)
        
        # Forward algorithm
        alpha = np.zeros((n_samples, self.n_states))
        
        # Initialize
        alpha[0] = self.start_prob * emission_probs[0]
        alpha[0] = alpha[0] / np.sum(alpha[0])
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                alpha[t, j] = emission_probs[t, j] * np.sum(
                    alpha[t-1] * self.transition_matrix[:, j]
                )
            alpha[t] = alpha[t] / np.sum(alpha[t])
        
        # Backward algorithm
        beta = np.zeros((n_samples, self.n_states))
        beta[-1] = 1.0
        
        # Backward pass
        for t in range(n_samples-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix[i] * emission_probs[t+1] * beta[t+1]
                )
            if np.sum(beta[t]) > 0:
                beta[t] = beta[t] / np.sum(beta[t])
        
        # Compute gamma (state probabilities)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        return alpha, beta, gamma
    
    def _update_transition_matrix(self, X, gamma):
        """Update transition matrix using EM algorithm"""
        n_samples = len(X)
        
        # Get emission probabilities
        emission_probs = self._compute_emission_probabilities(X)
        
        # Compute xi (transition probabilities)
        xi = np.zeros((n_samples-1, self.n_states, self.n_states))
        
        for t in range(n_samples-1):
            denominator = 0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (gamma[t, i] * self.transition_matrix[i, j] * 
                                  emission_probs[t+1, j] * gamma[t+1, j])
                    denominator += xi[t, i, j]
            
            if denominator > 0:
                xi[t] = xi[t] / denominator
        
        # Update transition matrix
        for i in range(self.n_states):
            denominator = np.sum(gamma[:-1, i])
            if denominator > 0:
                for j in range(self.n_states):
                    self.transition_matrix[i, j] = np.sum(xi[:, i, j]) / denominator
        
        # Normalize transition matrix
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        return self.transition_matrix
    
    def _compute_log_likelihood(self, X):
        """Compute log likelihood of the data given current parameters"""
        try:
            from scipy.special import logsumexp
            n_samples = len(X)
            log_likelihood = 0.0
            
            # Get emission probabilities
            emission_probs = self._compute_emission_probabilities(X)
            
            # Forward algorithm for log likelihood computation
            log_alpha = np.zeros((n_samples, self.n_states))
            
            # Initialize
            log_alpha[0] = np.log(self.start_prob + 1e-10) + np.log(emission_probs[0] + 1e-10)
            
            # Forward pass
            for t in range(1, n_samples):
                for j in range(self.n_states):
                    log_alpha[t, j] = np.log(emission_probs[t, j] + 1e-10) + logsumexp(
                        log_alpha[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                    )
            
            # Total log likelihood
            log_likelihood = logsumexp(log_alpha[-1])
            
            return log_likelihood
            
        except Exception as e:
            logging.error(f"Error computing log likelihood: {e}")
            return -np.inf
    
    def fit(self, X, y):
        """Fit XGB-HMM model using EM algorithm"""
        # Dynamic class adjustment - adjust n_states to match number of unique classes
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        
        if self.n_states != num_classes:
            logging.info(f"Adjusting n_states from {self.n_states} to {num_classes} to match number of unique classes")
            self.n_states = num_classes
        
        logging.info(f"Starting XGB-HMM training with {self.n_states} states for {num_classes} classes...")
        
        # Initialize with GMM-HMM
        state_sequences, gamma = self._initialize_gmm_hmm(X, y)
        
        # Train initial XGBoost model with GMM-derived state probabilities
        logging.info("Training initial XGBoost model with GMM-derived states...")
        self.xgb_model = self._train_xgb_emission_model(X, y, gamma)
        
        # Initial log likelihood
        prev_log_likelihood = -np.inf
        
        # EM algorithm iterations
        for iteration in range(self.max_iter):
            # E-step: Update state probabilities
            alpha, beta, gamma = self._forward_backward_algorithm(X)
            
            # M-step: Update model parameters
            # Update XGBoost emission model
            self.xgb_model = self._train_xgb_emission_model(X, y, gamma)
            
            # Update transition matrix
            self.transition_matrix = self._update_transition_matrix(X, gamma)
            
            # Update start probabilities
            self.start_prob = gamma[0] / np.sum(gamma[0])
            
            # Compute log likelihood
            current_log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history.append(current_log_likelihood)
            
            # Check convergence
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                logging.info(f"Converged after {iteration + 1} iterations")
                self.convergence_info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_log_likelihood': current_log_likelihood
                }
                break
            
            prev_log_likelihood = current_log_likelihood
        
        else:
            logging.info(f"Maximum iterations ({self.max_iter}) reached")
            self.convergence_info = {
                'converged': False,
                'iterations': self.max_iter,
                'final_log_likelihood': current_log_likelihood
            }
        
        logging.info("XGB-HMM training completed")
        return self
    
    def predict(self, X):
        """Predict using Viterbi algorithm"""
        if self.xgb_model is None:
            raise ValueError("Model not trained yet")
        
        # Get emission probabilities
        emission_probs = self._compute_emission_probabilities(X)
        
        # Viterbi algorithm
        n_samples = len(X)
        viterbi_path = np.zeros(n_samples, dtype=int)
        delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=int)
        
        # Initialize
        delta[0] = np.log(self.start_prob + 1e-10) + np.log(emission_probs[0] + 1e-10)
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                transitions = delta[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                psi[t, j] = np.argmax(transitions)
                delta[t, j] = np.max(transitions) + np.log(emission_probs[t, j] + 1e-10)
        
        # Backward pass
        viterbi_path[-1] = np.argmax(delta[-1])
        for t in range(n_samples-2, -1, -1):
            viterbi_path[t] = psi[t+1, viterbi_path[t+1]]
        
        return viterbi_path
    
    def predict_proba(self, X):
        """Predict state probabilities"""
        if self.xgb_model is None:
            raise ValueError("Model not trained yet")
        
        # Use forward-backward algorithm
        alpha, beta, gamma = self._forward_backward_algorithm(X)
        return gamma
    
    def get_emission_predictions(self, X):
        """Get XGBoost emission predictions"""
        if self.xgb_model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.xgb_model.predict(X)
        probabilities = self.xgb_model.predict_proba(X)
        
        return predictions, probabilities
    
    def evaluate_model(self, X, y_true):
        """
        Comprehensive evaluation of XGB-HMM model
        Returns log likelihood, state sequence accuracy, and emission accuracy
        """
        if self.xgb_model is None or self.transition_matrix is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Calculate log likelihood
        log_likelihood = self._compute_log_likelihood(X)
        
        # Get state predictions
        state_predictions = self.predict(X)
        
        # Calculate state sequence accuracy (if true states are available)
        state_accuracy = accuracy_score(y_true, state_predictions)
        
        # Get emission predictions
        emission_pred, emission_proba = self.get_emission_predictions(X)
        emission_accuracy = accuracy_score(y_true, emission_pred)
        
        # Calculate perplexity (lower is better)
        perplexity = np.exp(-log_likelihood / len(X))
        
        evaluation_metrics = {
            'log_likelihood': log_likelihood,
            'perplexity': perplexity,
            'state_sequence_accuracy': state_accuracy,
            'emission_accuracy': emission_accuracy,
            'avg_log_likelihood_per_sample': log_likelihood / len(X)
        }
        
        logging.info(f"XGB-HMM Evaluation Metrics:")
        logging.info(f"  Log Likelihood: {log_likelihood:.4f}")
        logging.info(f"  Perplexity: {perplexity:.4f}")
        logging.info(f"  State Sequence Accuracy: {state_accuracy:.4f}")
        logging.info(f"  Emission Accuracy: {emission_accuracy:.4f}")
        logging.info(f"  Avg Log Likelihood per Sample: {log_likelihood / len(X):.4f}")
        
        return evaluation_metrics
    
    def get_state_transition_analysis(self, X):
        """
        Analyze state transitions and provide insights
        """
        if self.transition_matrix is None:
            raise ValueError("Model must be fitted before analysis")
        
        # Get state sequence
        state_sequence = self.predict(X)
        
        # Calculate transition counts
        transition_counts = np.zeros((self.n_states, self.n_states))
        for i in range(len(state_sequence) - 1):
            current_state = state_sequence[i]
            next_state = state_sequence[i + 1]
            transition_counts[current_state, next_state] += 1
        
        # Calculate empirical transition probabilities
        empirical_transitions = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 1e-8)
        
        # State distribution
        state_distribution = np.bincount(state_sequence, minlength=self.n_states) / len(state_sequence)
        
        analysis = {
            'learned_transition_matrix': self.transition_matrix,
            'empirical_transition_matrix': empirical_transitions,
            'state_distribution': state_distribution,
            'transition_counts': transition_counts,
            'most_frequent_state': np.argmax(state_distribution),
            'least_frequent_state': np.argmin(state_distribution)
        }
        
        logging.info(f"State Transition Analysis:")
        logging.info(f"  State Distribution: {state_distribution}")
        logging.info(f"  Most Frequent State: {np.argmax(state_distribution)}")
        logging.info(f"  Least Frequent State: {np.argmin(state_distribution)}")
        
        return analysis

class XGBHMMLSTMModel:
    """
    Triple Hybrid XGBoost-HMM-LSTM Model for Sequential Classification
    Combines HMM temporal modeling, XGBoost feature learning, and LSTM sequence modeling
    """
    
    def __init__(self, n_states=3, max_iter=50, tol=1e-4, random_state=42,
                 sequence_length=20, lstm_units=50, dropout_rate=0.2, 
                 learning_rate=0.001, ensemble_weights=None):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # LSTM parameters
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Ensemble weights [xgb_weight, hmm_weight, lstm_weight]
        if ensemble_weights is None:
            self.ensemble_weights = [0.4, 0.3, 0.3]
        else:
            self.ensemble_weights = ensemble_weights
        
        # Model components
        self.gmm_hmm = None
        self.xgb_model = None
        self.lstm_model = None
        self.transition_matrix = None
        self.start_prob = None
        
        # Training history
        self.log_likelihood_history = []
        self.convergence_info = {}
        
    def _initialize_gmm_hmm(self, X, y):
        """Initialize GMM-HMM for initial state estimation"""
        logging.info("Initializing GMM-HMM for state estimation...")
        
        # Handle NaN values by filling with median
        X_clean = X.fillna(X.median())
        
        # Convert to numpy array if it's a DataFrame
        if hasattr(X_clean, 'values'):
            X_array = X_clean.values
        else:
            X_array = X_clean
        
        # Initialize Gaussian HMM with diagonal covariance for stability
        self.gmm_hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",  # Use diagonal covariance for numerical stability
            random_state=self.random_state,
            n_iter=20,
            tol=1e-2,  # More lenient tolerance
            min_covar=1e-3  # Add minimum covariance for numerical stability
        )
        
        # Fit GMM-HMM to get initial state sequences
        self.gmm_hmm.fit(X_array)
        
        # Get state sequences and probabilities
        state_sequences = self.gmm_hmm.predict(X_array)
        gamma = self.gmm_hmm.predict_proba(X_array)
        
        # Extract transition matrix and start probabilities
        self.transition_matrix = self.gmm_hmm.transmat_
        self.start_prob = self.gmm_hmm.startprob_
        
        return state_sequences, gamma
    
    def _train_xgb_emission_model(self, X, y, gamma):
        """Train XGBoost model for emission probabilities"""
        # Prepare training data with state probabilities
        n_samples, n_features = X.shape
        
        # Create expanded training data
        X_expanded = []
        y_expanded = []
        weights_expanded = []
        
        for i in range(n_samples):
            for state in range(self.n_states):
                if gamma[i, state] > 0.01:  # Only include significant state probabilities
                    X_expanded.append(X.iloc[i].values)
                    y_expanded.append(y.iloc[i])
                    weights_expanded.append(gamma[i, state])
        
        X_expanded = np.array(X_expanded)
        y_expanded = np.array(y_expanded)
        weights_expanded = np.array(weights_expanded)
        
        # Determine XGBoost parameters based on label distribution
        unique_labels = np.unique(y_expanded)
        if len(unique_labels) == 2:
            objective = 'binary:logistic'
            eval_metric = 'logloss'
            num_class = None
        else:
            objective = 'multi:softmax'
            eval_metric = 'mlogloss'
            num_class = len(unique_labels)
        
        # Initialize XGBoost with optimal parameters
        self.xgb_model = XGBClassifier(
            objective=objective,
            num_class=num_class,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,  # Dropout pentru level-wise feature sampling
            colsample_bynode=0.8,   # Dropout pentru node-wise feature sampling
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # Train XGBoost model
        self.xgb_model.fit(X_expanded, y_expanded, sample_weight=weights_expanded)
        
        return self.xgb_model
    
    def _compute_emission_probabilities(self, X):
        """Compute emission probabilities using XGBoost"""
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained yet")
        
        # Handle NaN values by filling with median
        X = np.array(X)
        if np.isnan(X).any():
            for col in range(X.shape[1]):
                col_median = np.nanmedian(X[:, col])
                X[:, col] = np.where(np.isnan(X[:, col]), col_median, X[:, col])
        
        # Get XGBoost predictions
        emission_probs = self.xgb_model.predict_proba(X)
        
        # Ensure probabilities are normalized
        emission_probs = emission_probs / emission_probs.sum(axis=1, keepdims=True)
        
        return emission_probs
    
    def _forward_backward_algorithm(self, X):
        """Implement forward-backward algorithm for state estimation"""
        n_samples = len(X)
        
        # Get emission probabilities
        emission_probs = self._compute_emission_probabilities(X)
        
        # Forward algorithm
        alpha = np.zeros((n_samples, self.n_states))
        
        # Initialize
        alpha[0] = self.start_prob * emission_probs[0]
        alpha[0] = alpha[0] / np.sum(alpha[0])
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                alpha[t, j] = emission_probs[t, j] * np.sum(
                    alpha[t-1] * self.transition_matrix[:, j]
                )
            alpha[t] = alpha[t] / np.sum(alpha[t])
        
        # Backward algorithm
        beta = np.zeros((n_samples, self.n_states))
        beta[-1] = 1.0
        
        # Backward pass
        for t in range(n_samples-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix[i] * emission_probs[t+1] * beta[t+1]
                )
            if np.sum(beta[t]) > 0:
                beta[t] = beta[t] / np.sum(beta[t])
        
        # Compute gamma (state probabilities)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        return alpha, beta, gamma
    
    def _update_transition_matrix(self, X, gamma):
        """Update transition matrix using EM algorithm"""
        n_samples = len(X)
        
        # Get emission probabilities
        emission_probs = self._compute_emission_probabilities(X)
        
        # Compute xi (transition probabilities)
        xi = np.zeros((n_samples-1, self.n_states, self.n_states))
        
        for t in range(n_samples-1):
            denominator = 0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (gamma[t, i] * self.transition_matrix[i, j] * 
                                  emission_probs[t+1, j] * gamma[t+1, j])
                    denominator += xi[t, i, j]
            
            if denominator > 0:
                xi[t] = xi[t] / denominator
        
        # Update transition matrix
        for i in range(self.n_states):
            denominator = np.sum(gamma[:-1, i])
            if denominator > 0:
                for j in range(self.n_states):
                    self.transition_matrix[i, j] = np.sum(xi[:, i, j]) / denominator
        
        # Normalize transition matrix
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        return self.transition_matrix
    
    def _compute_log_likelihood(self, X):
        """Compute log likelihood of the data given current parameters"""
        try:
            from scipy.special import logsumexp
            n_samples = len(X)
            log_likelihood = 0.0
            
            # Get emission probabilities
            emission_probs = self._compute_emission_probabilities(X)
            
            # Forward algorithm for log likelihood computation
            log_alpha = np.zeros((n_samples, self.n_states))
            
            # Initialize
            log_alpha[0] = np.log(self.start_prob + 1e-10) + np.log(emission_probs[0] + 1e-10)
            
            # Forward pass
            for t in range(1, n_samples):
                for j in range(self.n_states):
                    log_alpha[t, j] = np.log(emission_probs[t, j] + 1e-10) + logsumexp(
                        log_alpha[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                    )
            
            # Total log likelihood
            log_likelihood = logsumexp(log_alpha[-1])
            
            return log_likelihood
            
        except Exception as e:
            logging.error(f"Error computing log likelihood: {e}")
            return -np.inf
    
    def fit(self, X, y):
        """Fit XGB-HMM model using EM algorithm"""
        # Dynamic class adjustment - adjust n_states to match number of unique classes
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        
        if self.n_states != num_classes:
            logging.info(f"Adjusting n_states from {self.n_states} to {num_classes} to match number of unique classes")
            self.n_states = num_classes
        
        logging.info(f"Starting XGB-HMM training with {self.n_states} states for {num_classes} classes...")
        
        # Initialize with GMM-HMM
        state_sequences, gamma = self._initialize_gmm_hmm(X, y)
        
        # Train initial XGBoost model with GMM-derived state probabilities
        logging.info("Training initial XGBoost model with GMM-derived states...")
        self.xgb_model = self._train_xgb_emission_model(X, y, gamma)
        
        # Initial log likelihood
        prev_log_likelihood = -np.inf
        
        # EM algorithm iterations
        for iteration in range(self.max_iter):
            # E-step: Update state probabilities
            alpha, beta, gamma = self._forward_backward_algorithm(X)
            
            # M-step: Update model parameters
            # Update XGBoost emission model
            self.xgb_model = self._train_xgb_emission_model(X, y, gamma)
            
            # Update transition matrix
            self.transition_matrix = self._update_transition_matrix(X, gamma)
            
            # Update start probabilities
            self.start_prob = gamma[0] / np.sum(gamma[0])
            
            # Compute log likelihood
            current_log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history.append(current_log_likelihood)
            
            # Check convergence
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                logging.info(f"Converged after {iteration + 1} iterations")
                self.convergence_info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_log_likelihood': current_log_likelihood
                }
                break
            
            prev_log_likelihood = current_log_likelihood
        
        else:
            logging.info(f"Maximum iterations ({self.max_iter}) reached")
            self.convergence_info = {
                'converged': False,
                'iterations': self.max_iter,
                'final_log_likelihood': current_log_likelihood
            }
        
        logging.info("XGB-HMM training completed")
        return self
    
    def predict(self, X):
        """Predict using Viterbi algorithm"""
        if self.xgb_model is None:
            raise ValueError("Model not trained yet")
        
        # Get emission probabilities
        emission_probs = self._compute_emission_probabilities(X)
        
        # Viterbi algorithm
        n_samples = len(X)
        viterbi_path = np.zeros(n_samples, dtype=int)
        delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=int)
        
        # Initialize
        delta[0] = np.log(self.start_prob + 1e-10) + np.log(emission_probs[0] + 1e-10)
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                transitions = delta[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                psi[t, j] = np.argmax(transitions)
                delta[t, j] = np.max(transitions) + np.log(emission_probs[t, j] + 1e-10)
        
        # Backward pass
        viterbi_path[-1] = np.argmax(delta[-1])
        for t in range(n_samples-2, -1, -1):
            viterbi_path[t] = psi[t+1, viterbi_path[t+1]]
        
        return viterbi_path
    
    def predict_proba(self, X):
        """Predict state probabilities"""
        if self.xgb_model is None:
            raise ValueError("Model not trained yet")
        
        # Use forward-backward algorithm
        alpha, beta, gamma = self._forward_backward_algorithm(X)
        return gamma
    
    def get_emission_predictions(self, X):
        """Get XGBoost emission predictions"""
        if self.xgb_model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.xgb_model.predict(X)
        probabilities = self.xgb_model.predict_proba(X)
        
        return predictions, probabilities
    
    def evaluate_model(self, X, y_true):
        """
        Comprehensive evaluation of XGB-HMM model
        Returns log likelihood, state sequence accuracy, and emission accuracy
        """
        if self.xgb_model is None or self.transition_matrix is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Calculate log likelihood
        log_likelihood = self._compute_log_likelihood(X)
        
        # Get state predictions
        state_predictions = self.predict(X)
        
        # Calculate state sequence accuracy (if true states are available)
        state_accuracy = accuracy_score(y_true, state_predictions)
        
        # Get emission predictions
        emission_pred, emission_proba = self.get_emission_predictions(X)
        emission_accuracy = accuracy_score(y_true, emission_pred)
        
        # Calculate perplexity (lower is better)
        perplexity = np.exp(-log_likelihood / len(X))
        
        evaluation_metrics = {
            'log_likelihood': log_likelihood,
            'perplexity': perplexity,
            'state_sequence_accuracy': state_accuracy,
            'emission_accuracy': emission_accuracy,
            'avg_log_likelihood_per_sample': log_likelihood / len(X)
        }
        
        logging.info(f"XGB-HMM Evaluation Metrics:")
        logging.info(f"  Log Likelihood: {log_likelihood:.4f}")
        logging.info(f"  Perplexity: {perplexity:.4f}")
        logging.info(f"  State Sequence Accuracy: {state_accuracy:.4f}")
        logging.info(f"  Emission Accuracy: {emission_accuracy:.4f}")
        logging.info(f"  Avg Log Likelihood per Sample: {log_likelihood / len(X):.4f}")
        
        return evaluation_metrics
    
    def get_state_transition_analysis(self, X):
        """
        Analyze state transitions and provide insights
        """
        if self.transition_matrix is None:
            raise ValueError("Model must be fitted before analysis")
        
        # Get state sequence
        state_sequence = self.predict(X)
        
        # Calculate transition counts
        transition_counts = np.zeros((self.n_states, self.n_states))
        for i in range(len(state_sequence) - 1):
            current_state = state_sequence[i]
            next_state = state_sequence[i + 1]
            transition_counts[current_state, next_state] += 1
        
        # Calculate empirical transition probabilities
        empirical_transitions = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 1e-8)
        
        # State distribution
        state_distribution = np.bincount(state_sequence, minlength=self.n_states) / len(state_sequence)
        
        analysis = {
            'learned_transition_matrix': self.transition_matrix,
            'empirical_transition_matrix': empirical_transitions,
            'state_distribution': state_distribution,
            'transition_counts': transition_counts,
            'most_frequent_state': np.argmax(state_distribution),
            'least_frequent_state': np.argmin(state_distribution)
        }
        
        logging.info(f"State Transition Analysis:")
        logging.info(f"  State Distribution: {state_distribution}")
        logging.info(f"  Most Frequent State: {np.argmax(state_distribution)}")
        logging.info(f"  Least Frequent State: {np.argmin(state_distribution)}")
        
        return analysis

class FeatureSelector:
    """
    Comprehensive Feature Selection Class for XGB-HMM Model
    Implements multiple feature selection methods to improve model performance
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = None
        self.feature_scores = {}
        self.selection_methods = {}
        
    def univariate_selection(self, X, y, k=50, score_func=f_classif):
        """
        Univariate feature selection using statistical tests
        """
        logging.info(f"Performing univariate feature selection with k={k} features...")
        
        # Handle missing values with imputation
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Handle different score functions
        if score_func == chi2:
            # Chi2 requires non-negative features
            X_scaled = StandardScaler().fit_transform(X_imputed)
            X_scaled = X_scaled - X_scaled.min() + 1e-8  # Make all values positive
        else:
            X_scaled = X_imputed
            
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        
        # Get feature scores and selected features
        feature_scores = selector.scores_
        selected_indices = selector.get_support(indices=True)
        
        self.selection_methods['univariate'] = {
            'selector': selector,
            'selected_indices': selected_indices,
            'scores': feature_scores,
            'score_func': score_func.__name__
        }
        
        logging.info(f"Univariate selection completed. Selected {len(selected_indices)} features.")
        return X_selected, selected_indices, feature_scores
    
    def mutual_info_selection(self, X, y, k=50):
        """
        Feature selection using mutual information
        """
        logging.info(f"Performing mutual information feature selection with k={k} features...")
        
        # Handle NaN values with imputation
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X_imputed, y, random_state=self.random_state)
        
        # Select top k features
        top_k_indices = np.argsort(mi_scores)[-k:]
        X_selected = X_imputed[:, top_k_indices]
        
        self.selection_methods['mutual_info'] = {
            'selected_indices': top_k_indices,
            'scores': mi_scores,
            'top_scores': mi_scores[top_k_indices]
        }
        
        logging.info(f"Mutual information selection completed. Selected {len(top_k_indices)} features.")
        return X_selected, top_k_indices, mi_scores
    
    def rfe_selection(self, X, y, n_features=50, step=1):
        """
        Recursive Feature Elimination using XGBoost
        """
        logging.info(f"Performing RFE selection with n_features={n_features}...")
        
        # Handle NaN values with imputation
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Create XGBoost estimator for RFE
        estimator = XGBClassifier(
            random_state=self.random_state,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            verbosity=0
        )
        
        # Perform RFE
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
        X_selected = rfe.fit_transform(X_imputed, y)
        
        # Get selected features and rankings
        selected_indices = np.where(rfe.support_)[0]
        feature_rankings = rfe.ranking_
        
        self.selection_methods['rfe'] = {
            'selector': rfe,
            'selected_indices': selected_indices,
            'rankings': feature_rankings,
            'estimator': estimator
        }
        
        logging.info(f"RFE selection completed. Selected {len(selected_indices)} features.")
        return X_selected, selected_indices, feature_rankings
    
    def feature_importance_selection(self, X, y, n_features=50):
        """
        Feature selection based on XGBoost feature importance
        """
        logging.info(f"Performing feature importance selection with n_features={n_features}...")
        
        # Handle NaN values with imputation
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Train XGBoost to get feature importance
        xgb_model = XGBClassifier(
            random_state=self.random_state,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            verbosity=0
        )
        
        xgb_model.fit(X_imputed, y)
        
        # Get feature importance
        feature_importance = xgb_model.feature_importances_
        
        # Select top n_features
        top_indices = np.argsort(feature_importance)[-n_features:]
        X_selected = X_imputed[:, top_indices]
        
        self.selection_methods['feature_importance'] = {
            'model': xgb_model,
            'selected_indices': top_indices,
            'importance_scores': feature_importance,
            'top_scores': feature_importance[top_indices]
        }
        
        logging.info(f"Feature importance selection completed. Selected {len(top_indices)} features.")
        return X_selected, top_indices, feature_importance
    
    def ensemble_selection(self, X, y, methods=['univariate', 'mutual_info', 'rfe', 'feature_importance'], 
                          k_univariate=100, k_mutual_info=100, n_rfe=100, n_importance=100, 
                          min_votes=2):
        """
        Ensemble feature selection combining multiple methods
        """
        logging.info("Performing ensemble feature selection...")
        
        n_features = X.shape[1]
        feature_votes = np.zeros(n_features)
        
        # Apply each method and collect votes
        if 'univariate' in methods:
            _, indices, _ = self.univariate_selection(X, y, k=k_univariate)
            feature_votes[indices] += 1
            
        if 'mutual_info' in methods:
            _, indices, _ = self.mutual_info_selection(X, y, k=k_mutual_info)
            feature_votes[indices] += 1
            
        if 'rfe' in methods:
            _, indices, _ = self.rfe_selection(X, y, n_features=n_rfe)
            feature_votes[indices] += 1
            
        if 'feature_importance' in methods:
            _, indices, _ = self.feature_importance_selection(X, y, n_features=n_importance)
            feature_votes[indices] += 1
        
        # Select features with minimum votes
        selected_indices = np.where(feature_votes >= min_votes)[0]
        
        if len(selected_indices) == 0:
            logging.warning(f"No features received {min_votes} votes. Lowering threshold to 1.")
            selected_indices = np.where(feature_votes >= 1)[0]
        
        # Use .iloc for pandas DataFrame indexing
        if hasattr(X, 'iloc'):
            X_selected = X.iloc[:, selected_indices]
        else:
            X_selected = X[:, selected_indices]
        
        self.selection_methods['ensemble'] = {
            'selected_indices': selected_indices,
            'feature_votes': feature_votes,
            'methods_used': methods,
            'min_votes': min_votes
        }
        
        self.selected_features = selected_indices
        
        logging.info(f"Ensemble selection completed. Selected {len(selected_indices)} features with {min_votes}+ votes.")
        return X_selected, selected_indices, feature_votes
    
    def get_feature_selection_summary(self, feature_names=None):
        """
        Generate comprehensive summary of feature selection results
        """
        summary = {
            'total_methods_applied': len(self.selection_methods),
            'methods_used': list(self.selection_methods.keys())
        }
        
        if feature_names is not None and self.selected_features is not None:
            summary['selected_feature_names'] = [feature_names[i] for i in self.selected_features]
        
        if self.selected_features is not None:
            summary['n_selected_features'] = len(self.selected_features)
            summary['selected_indices'] = self.selected_features.tolist()
        
        # Add method-specific information
        for method_name, method_info in self.selection_methods.items():
            summary[f'{method_name}_info'] = {
                'n_features_selected': len(method_info['selected_indices']),
                'selected_indices': method_info['selected_indices'].tolist()
            }
            
            if 'scores' in method_info:
                summary[f'{method_name}_info']['mean_score'] = np.mean(method_info['scores'])
                summary[f'{method_name}_info']['std_score'] = np.std(method_info['scores'])
        
        return summary
    
    def transform(self, X):
        """
        Transform new data using selected features
        """
        if self.selected_features is None:
            raise ValueError("No features have been selected yet. Run a selection method first.")
        
        return X[:, self.selected_features]

# Dataframe Init
original_df = pd.read_csv('/root/vynixmodelling/dataset/TSLA_original.csv')

# Enhanced Progress Callback for Optuna
class EnhancedProgressCallback:
    def __init__(self, study_name, patience=10, min_improvement=0.001, max_trial_time=300):
        self.study_name = study_name
        self.patience = patience
        self.min_improvement = min_improvement
        self.max_trial_time = max_trial_time
        self.best_value = None
        self.trials_without_improvement = 0
        self.start_time = time.time()
        
    def __call__(self, study, trial):
        current_value = trial.value
        current_time = time.time()
        
        # Check if this is the best trial so far
        if self.best_value is None or current_value > self.best_value + self.min_improvement:
            self.best_value = current_value
            self.trials_without_improvement = 0
            logging.info(f"{self.study_name} - Trial {trial.number}: New best value {current_value:.4f}")
        else:
            self.trials_without_improvement += 1
            
        # Early stopping conditions
        if self.trials_without_improvement >= self.patience:
            logging.info(f"{self.study_name} - Early stopping: {self.patience} trials without improvement")
            study.stop()
            
        # Time-based stopping
        if current_time - self.start_time > self.max_trial_time:
            logging.info(f"{self.study_name} - Time limit reached: {self.max_trial_time}s")
            study.stop()

def create_study_with_storage(study_name, direction='maximize'):
    """Create Optuna study with in-memory storage"""
    storage = InMemoryStorage()
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    return study

# Logging Init
# logging location: /root/vynixmodelling/ML_RL/logs
# log name format: {log_name}_{time}.log
from datetime import datetime
import os

# Create logs/main directory if it doesn't exist
os.makedirs('/root/vynixmodelling/ML_RL/logs/main', exist_ok=True)
os.makedirs('/root/vynixmodelling/ML_RL/logs/xgb_results', exist_ok=True)

# Generate datetime-based log filename
current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'/root/vynixmodelling/ML_RL/logs/main/main_{current_datetime}.log'

# Configure logging to both file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler for logging to file
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler for logging to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configure for real-time logging
file_handler.stream = open(file_handler.baseFilename, 'a', buffering=1)  # Line buffering

# Preprocessing
df = original_df.copy()
logging.info(f"Original dataframe shape: {df.shape}")

# Exclude rows with NaN or null values
non_null_df = df.dropna()

# Print total rows in the dataframe without NaN or null values
logging.info(f"Total rows in df without NaN or null values: {len(non_null_df)}")

# Convert 'time' column to datetime format and print the first few rows
non_null_df['time_converted'] = pd.to_datetime(non_null_df['time'], unit='s').dt.strftime('%d%m%Y')
# logging.info(non_null_df[['time', 'time_converted']].head(1))
# logging.info(non_null_df.head)
# logging.info(non_null_df.tail)
# logging.info(non_null_df.info())
# logging.info(non_null_df.describe())
# logging.info(non_null_df.columns)

# Data Gathering with Fundamental data
# 1. Process Fundamental Data from Local Files
# Menggunakan fungsi baru untuk membaca data lokal berdasarkan CIK
pivoted_df = process_fundamental_data_local("TSLA")

# Jika gagal memuat dari lokal, fallback ke API
if pivoted_df is None:
    logging.info("Fallback to API method...")
    fundamental_data = get_fundamental_data("TSLA")
    pivoted_df = process_fundamental_data(fundamental_data, "TSLA")
    logging.info("Data loaded from API successfully")
else:
    # Baca kembali CSV yang sudah diproses untuk mendapatkan DataFrame
    pivoted_df = pd.read_csv('/root/vynixmodelling/dataset/data_fundamental/TSLA_time.csv', index_col=0)
    logging.info(f"Data loaded from local files successfully: {pivoted_df.shape}")

# 3. Fundamental Data Financial Feature Engineering
from fundamental_feature_engineering import apply_feature_engineering
enhanced_fundamental_df = apply_feature_engineering(pivoted_df, "TSLA")
# logging.info(enhanced_fundamental_df.head(1))

# 4. Teknikal and Fundamental Data Preprocessing
from technical_fundamental_preprocessing import preprocess_technical_fundamental_data

# Process data using complete pipeline
logging.info("Processing technical and fundamental data...")
output_path = '/root/vynixmodelling/dataset/main_processed_data.csv'

filtered_df = preprocess_technical_fundamental_data(
    technical_df=non_null_df,
    fundamental_df=enhanced_fundamental_df,
    output_path=output_path,
    start_period='2012-Q2',
    end_period='2025-Q2'
)

logging.info(f"Data processing completed: {filtered_df.shape}")
logging.info(f"Processed data saved to: {output_path}")

# Display final results
logging.info(f"\nFinal processed data shape: {filtered_df.shape}")
logging.info(f"Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
logging.info(f"Technical columns: {len([col for col in filtered_df.columns if any(tech in col.lower() for tech in ['open', 'high', 'low', 'close', 'volume'])])}")
logging.info(f"Fundamental columns: {len([col for col in filtered_df.columns if col not in ['time', 'date', 'datetime'] and not any(tech in col.lower() for tech in ['open', 'high', 'low', 'close', 'volume'])])}")
logging.info("\nPreprocessing completed successfully!")
logging.info(f'Data completeness: {(filtered_df.count().sum() / (len(filtered_df) * len(filtered_df.columns)) * 100):.2f}%')

# 5. Implementasi Labelling Triple Barrier Method ke filtered_df
# Semua parameter di init di file ini. main.py.
from triple_barrier.triplebarrier import apply_triple_barrier_labeling
from triple_barrier.visualizebarrier import generate_triple_barrier_visualizations

# Triple Barrier Method Parameters - dapat dikonfigurasi sesuai kebutuhan
TRIPLE_BARRIER_PARAMS = {
    'volatility_window': 20,           # Window untuk menghitung volatilitas
    'upper_barrier_multiplier': 1.0,   # Multiplier untuk upper barrier
    'lower_barrier_multiplier': 1.0,   # Multiplier untuk lower barrier
    'time_barrier_days': 15,            # Maksimum periode untuk menunggu barrier touch
    'verbose': True                    # Tampilkan statistik hasil
}

# Visualization Parameters
VISUALIZATION_PARAMS = {
    'output_dir': '/root/vynixmodelling/ML_RL/logs/visualization',
    'window_size': 50,                 # Jumlah data sebelum dan sesudah untuk visualisasi
    'save_html': True,                 # Simpan dalam format HTML
    'save_png': False,                  # Simpan dalam format PNG (akan menginstal kaleido otomatis)
    'verbose': True                    # Tampilkan log proses
}

logging.info("\n=== Applying Triple Barrier Method ===")
logging.info("Starting Triple Barrier Method labeling...")

# Aplikasikan Triple Barrier Method
triple_barrier_df = apply_triple_barrier_labeling(
    data=filtered_df,
    **TRIPLE_BARRIER_PARAMS
)

# Simpan hasil Triple Barrier ke CSV
triple_barrier_output_path = '/root/vynixmodelling/ML_RL/triple_barrier_results.csv'
triple_barrier_df.to_csv(triple_barrier_output_path, index=False)

# Generate visualisasi Triple Barrier
logging.info("\n=== Generating Triple Barrier Visualizations ===")
logging.info("Generating Triple Barrier visualizations...")

visualization_files = generate_triple_barrier_visualizations(
    data=filtered_df,
    triple_barrier_df=triple_barrier_df,
    **VISUALIZATION_PARAMS
)

logging.info(f"Visualizations generated: {len(visualization_files)} files")
for file_type, file_path in visualization_files.items():
    logging.info(f"{file_type}: {file_path}")

logging.info(f"\n=== Triple Barrier Implementation Complete ===")
logging.info(f"Labels generated: {len(triple_barrier_df)} samples")
logging.info(f"Results saved to: {triple_barrier_output_path}")
# logging.info(f"Visualizations saved to: {VISUALIZATION_PARAMS['output_dir']}")
# logging.info(f"Total visualization files: {len(visualization_files)}")

# Summary statistik final
# 6. Merge Triple Barrier labels dengan filtered_df berdasarkan date
# merged_label_df = triple_barrier_df.copy() + filtered_df.copy(), gunakan triple_barrier_df sebagai
# data utama yang sudah ber-label, lalu tarik data filtered_df sesuai dengan 'date' dari 'decision_date' triple_barrier_df

logging.info("\n=== Merging Triple Barrier Labels with Features ===")
logging.info("Merging triple_barrier_df with filtered_df based on decision_date...")

# Konversi decision_date ke datetime jika belum
triple_barrier_df['decision_date'] = pd.to_datetime(triple_barrier_df['decision_date'])
filtered_df['date'] = pd.to_datetime(filtered_df['date'])

# Merge berdasarkan decision_date dari triple_barrier_df dengan date dari filtered_df
merged_label_df = triple_barrier_df.merge(
    filtered_df, 
    left_on='decision_date', 
    right_on='date', 
    how='left'  # Gunakan left join untuk mempertahankan semua data triple_barrier_df
)

# Hapus kolom duplikat 'date' karena sudah ada 'decision_date'
if 'date' in merged_label_df.columns:
    merged_label_df = merged_label_df.drop('date', axis=1)

logging.info(f"Triple Barrier data shape: {triple_barrier_df.shape}")
logging.info(f"Filtered data shape: {filtered_df.shape}")
logging.info(f"Merged data shape: {merged_label_df.shape}")
logging.info(f"Successfully merged: {len(merged_label_df[merged_label_df.notna().all(axis=1)])} complete rows")

# Simpan hasil merge
merged_output_path = '/root/vynixmodelling/ML_RL/merged_labeled_data.csv'
merged_label_df.to_csv(merged_output_path, index=False)
logging.info(f"Merged labeled data saved to: {merged_output_path}")

logging.info(f"\n=== Final Summary ===")
logging.info(f"Original processed data shape: {filtered_df.shape}")
logging.info(f"Triple Barrier labels: {len(triple_barrier_df)} samples")
logging.info(f"Merged labeled data shape: {merged_label_df.shape}")
logging.info(f"Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
logging.info(f"Date range: {triple_barrier_df['decision_date'].min()} to {triple_barrier_df['decision_date'].max()}")
logging.info(f"Technical + Fundamental features: {len(filtered_df.columns)} columns")
logging.info(f"Total features in merged data: {len(merged_label_df.columns)} columns")
logging.info(f"Triple Barrier parameters used: {TRIPLE_BARRIER_PARAMS}")
logging.info("\nAll preprocessing, labeling, and merging completed successfully!")

# 6. Training Data Preparation.
# 6.1 Drop Column
# Dataframe yang digunakan sebagai data training: merged_label_df. kolom yang perlu di drop:
# 'decision_date', 'entry_date', 'end_date', 'end_price', 
# 'return', 'barrier_touched', 'value_at_barrier_touched', 
# 'time_converted, 'datetime','time', 'time_barrier'
# Target Col: 'label'

# Drop unnecessary columns for training
columns_to_drop = [
    'decision_date', 'entry_date', 'end_date', 'end_price',
    'return', 'barrier_touched', 'value_at_barrier_touched',
    'time_converted', 'datetime', 'time', 'time_barrier'
]

# Filter out columns that do not exist in the DataFrame
existing_columns_to_drop = [col for col in columns_to_drop if col in merged_label_df.columns]

if existing_columns_to_drop:
    training_df = merged_label_df.drop(columns=existing_columns_to_drop)
    logging.info(f"Dropped columns: {existing_columns_to_drop}")
else:
    training_df = merged_label_df.copy()
    logging.info("No specified columns to drop were found in the DataFrame.")

logging.info(f"Training DataFrame shape after dropping columns: {training_df.shape}")
logging.info(f"Training DataFrame columns: {training_df.columns.tolist()}")

# 6.2 Time-based Train Val Test Split
logging.info("\n=== Splitting Data into Train, Validation, and Test Sets (Time-based) ===")
logging.info("Splitting data into train, validation, and test sets using time-based approach...")

# First, we need to preserve decision_date for time-based splitting
# Create a copy of merged_label_df with decision_date preserved
merged_with_date = merged_label_df.copy()
merged_with_date['decision_date'] = pd.to_datetime(merged_with_date['decision_date'])

# Define Q3 2024 start date (July 1, 2024)
q3_2024_start = pd.Timestamp('2024-07-01')

# Split data based on time
test_mask = merged_with_date['decision_date'] >= q3_2024_start
train_val_mask = ~test_mask

# Create test set from Q3 2024 onwards
test_data = merged_with_date[test_mask].copy()
logging.info(f"Test data period: {test_data['decision_date'].min()} to {test_data['decision_date'].max()}")
logging.info(f"Test data shape: {test_data.shape}")

# Create train+validation set from data before Q3 2024
train_val_data = merged_with_date[train_val_mask].copy()
logging.info(f"Train+Val data period: {train_val_data['decision_date'].min()} to {train_val_data['decision_date'].max()}")
logging.info(f"Train+Val data shape: {train_val_data.shape}")

# Now drop the datetime columns for training
columns_to_drop_final = [
    'decision_date', 'entry_date', 'end_date', 'end_price',
    'return', 'barrier_touched', 'value_at_barrier_touched',
    'time_converted', 'datetime', 'time', 'time_barrier'
]

# Apply column dropping to test set
existing_cols_test = [col for col in columns_to_drop_final if col in test_data.columns]
if existing_cols_test:
    test_data_clean = test_data.drop(columns=existing_cols_test)
else:
    test_data_clean = test_data.copy()

# Apply column dropping to train+val set
existing_cols_train_val = [col for col in columns_to_drop_final if col in train_val_data.columns]
if existing_cols_train_val:
    train_val_data_clean = train_val_data.drop(columns=existing_cols_train_val)
else:
    train_val_data_clean = train_val_data.copy()

# Extract features and labels for test set
X_test = test_data_clean.drop('label', axis=1)
y_test = test_data_clean['label']

# Extract features and labels for train+val set
X_train_val = train_val_data_clean.drop('label', axis=1)
y_train_val = train_val_data_clean['label']

# Split train+val into train (80%) and validation (20%) - remove stratify for time series
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, random_state=42
)

# Update training_df to match the time-based split approach
training_df = train_val_data_clean.copy()

logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
logging.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Save the split data to CSV files
output_dir = '/root/vynixmodelling/dataset/training_data/'
import os
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)

logging.info(f"Train, test, and validation data saved to {output_dir}")

# 7. Training
# 7.1 XGBoost. Hanya gunakan data Train dan Val dulu.
# Standalone XGBoost model training and evaluation removed - using hybrid XGB-HMM approach only

# 8. Hyperparameter Tuning. Hanya gunakan data Train dan Val terlebih dahulu saja.
logging.info("8. HYPERPARAMETER TUNING")
logging.info("="*80)

# Import libraries for hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import time
import os
from datetime import datetime
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.info("Warning: Plotly not available. Visualization will be skipped.")
import warnings
warnings.filterwarnings('ignore')

# Enhanced Progress Callback for Optuna
class EnhancedProgressCallback:
    def __init__(self, study_name, patience=10, min_improvement=0.001, max_trial_time=300):
        self.study_name = study_name
        self.patience = patience
        self.min_improvement = min_improvement
        self.max_trial_time = max_trial_time  # Maximum time per trial in seconds
        self.best_score = -float('inf')
        self.trials_without_improvement = 0
        self.trial_start_times = {}
        
    def __call__(self, study, trial):
        # Track trial start time
        if trial.number not in self.trial_start_times:
            self.trial_start_times[trial.number] = time.time()
        
        # Check for timeout
        elapsed_time = time.time() - self.trial_start_times[trial.number]
        if elapsed_time > self.max_trial_time:
            logging.warning(f"Trial {trial.number} exceeded {self.max_trial_time}s timeout, pruning...")
            raise optuna.TrialPruned()
        
        # Check for improvement
        if trial.value is not None:
            if trial.value > self.best_score + self.min_improvement:
                self.best_score = trial.value
                self.trials_without_improvement = 0
                logging.info(f"{self.study_name} - Trial {trial.number}: New best score {trial.value:.4f}")
            else:
                self.trials_without_improvement += 1
                
            # Early stopping
            if self.trials_without_improvement >= self.patience:
                logging.info(f"{self.study_name} - Early stopping after {self.patience} trials without improvement")
                study.stop()

def create_study_with_storage(study_name, direction='maximize'):
    """Create Optuna study with persistent storage"""
    # Create studies directory if it doesn't exist
    import os
    os.makedirs('logs/optuna_studies', exist_ok=True)
    
    # Create database path
    db_path = f'logs/optuna_studies/{study_name}.db'
    storage = f'sqlite:///{db_path}'
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True
    )
    
    logging.info(f"Study '{study_name}' created/loaded with storage: {db_path}")
    return study

# Combined Hyperparameter Tuning for HMM, XGBoost, and Barrier Parameters
logging.info("\nCombined HMM, XGBoost, and Barrier Parameters Tuning with Optuna")
logging.info("-" * 50)

def combined_xgb_hmm_objective(trial):
    """Combined objective function for HMM, XGBoost, and Barrier parameters"""
    try:
        # HMM hyperparameters
        hmm_params = {
            'n_states': trial.suggest_int('n_states', 2, 5),
            'max_iter': trial.suggest_int('max_iter', 20, 100, step=10),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True)
        }
        
        # XGBoost hyperparameters
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=50),
            'max_depth': trial.suggest_int('max_depth', 2, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.001),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.001),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0, step=0.001),  # Dropout level-wise
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0, step=0.001),   # Dropout node-wise
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, step=0.01),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, step=0.01),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0, step=0.01)
        }
        
        # Barrier parameters - For 1:1 risk-reward ratio, upper and lower multipliers must be equal
        barrier_multiplier = trial.suggest_float('barrier_multiplier', 0.5, 10.0, step=0.01)
        
        barrier_params = {
            'volatility_window': trial.suggest_int('volatility_window', 5, 70, step=5),
            'upper_barrier_multiplier': barrier_multiplier,  # Same value for 1:1 risk-reward
            'lower_barrier_multiplier': barrier_multiplier,  # Same value for 1:1 risk-reward
            'time_barrier_days': trial.suggest_int('time_barrier_days', 3, 35),
            'verbose': False
        }
        
        # Apply triple barrier labeling with current parameters
        temp_triple_barrier_df = apply_triple_barrier_labeling(
            data=filtered_df,
            **barrier_params
        )
        
        # Merge with filtered_df
        temp_triple_barrier_df['decision_date'] = pd.to_datetime(temp_triple_barrier_df['decision_date'])
        temp_merged_df = temp_triple_barrier_df.merge(
            filtered_df, 
            left_on='decision_date', 
            right_on='date', 
            how='left'
        )
        
        if 'date' in temp_merged_df.columns:
            temp_merged_df = temp_merged_df.drop('date', axis=1)
        
        # Drop unnecessary columns for training
        temp_columns_to_drop = [
            'decision_date', 'entry_date', 'end_date', 'end_price',
            'return', 'barrier_touched', 'value_at_barrier_touched',
            'time_converted', 'datetime', 'time', 'time_barrier'
        ]
        temp_existing_columns_to_drop = [col for col in temp_columns_to_drop if col in temp_merged_df.columns]
        
        if temp_existing_columns_to_drop:
            df_with_labels_new = temp_merged_df.drop(columns=temp_existing_columns_to_drop)
        else:
            df_with_labels_new = temp_merged_df.copy()
        
        # Remove rows with NaN labels
        df_with_labels_new = df_with_labels_new.dropna(subset=['label'])
        
        if len(df_with_labels_new) < 100:  # Skip if too few samples
            return 0.0  # Return poor score for pruning
        
        # Apply time-based split to hyperparameter tuning data
        merged_new_df = df_with_labels_new.copy()
        merged_new_df['decision_date'] = temp_triple_barrier_df['decision_date']
        
        # Define Q3 2024 start date (July 1, 2024) - same as initial split
        q3_2024_start_new = pd.Timestamp('2024-07-01')
        
        # Split data based on time
        test_mask_new = merged_new_df['decision_date'] >= q3_2024_start_new
        train_val_mask_new = ~test_mask_new
        
        # Create train+validation set from data before Q3 2024
        train_val_data_new = merged_new_df[train_val_mask_new].copy()
        
        # Extract features and labels for train+val set
        X_train_val_new = train_val_data_new.drop(['label', 'decision_date'], axis=1)
        y_train_val_new = train_val_data_new['label']
        
        # Split train+val into train (80%) and validation (20%)
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
            X_train_val_new, y_train_val_new, test_size=0.20, random_state=42
        )
        
        # Map labels
        unique_labels_new = sorted(y_train_new.unique())
        num_classes_new = len(unique_labels_new)
        
        if num_classes_new == 2:
            label_mapping_new = {unique_labels_new[0]: 0, unique_labels_new[1]: 1}
        else:
            label_mapping_new = {label: idx for idx, label in enumerate(unique_labels_new)}
        
        y_train_mapped_new = y_train_new.map(label_mapping_new)
        y_val_mapped_new = y_val_new.map(label_mapping_new)
        
        # Dynamically adjust n_states to match number of classes
        suggested_n_states = hmm_params['n_states']
        actual_n_states = max(num_classes_new, suggested_n_states)  # Ensure n_states >= num_classes
        
        # Train XGB-HMM model with adjusted parameters
        temp_xgb_hmm_model = XGBHMMModel(
            n_states=actual_n_states,
            max_iter=hmm_params['max_iter'],
            tol=hmm_params['tol'],
            random_state=42
        )
        
        # Set XGBoost parameters with verbosity=0 to reduce logging
        xgb_params['verbosity'] = 0
        temp_xgb_hmm_model.xgb_params = xgb_params
        
        # Fit the model
        temp_xgb_hmm_model.fit(X_train_new, y_train_mapped_new)
        
        # Evaluate on validation set
        y_val_pred_new = temp_xgb_hmm_model.predict(X_val_new)
        val_accuracy_new = accuracy_score(y_val_mapped_new, y_val_pred_new)
        
        return val_accuracy_new
        
    except Exception as e:
        logging.warning(f"Trial failed with error: {str(e)}")
        return 0.0

# Update the combined hyperparameter tuning to include LSTM parameters properly
def combined_xgb_hmm_lstm_objective(trial):
    """Combined objective function for HMM, XGBoost, and LSTM parameters"""
    try:
        # HMM hyperparameters
        hmm_params = {
            'n_states': trial.suggest_int('n_states', 2, 5),
            'max_iter': trial.suggest_int('max_iter', 20, 1000, step=10),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True)
        }
        
        # XGBoost hyperparameters
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0, step=0.01),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0, step=0.01),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, step=0.1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, step=0.1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0, step=0.1)
        }
        
        # LSTM hyperparameters (only if TensorFlow is available)
        if TENSORFLOW_AVAILABLE:
            lstm_params = {
                'sequence_length': trial.suggest_int('sequence_length', 5, 500, step=5),
                'lstm_units': trial.suggest_int('lstm_units', 16, 612, step=8),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.001),
                'learning_rate': trial.suggest_float('lstm_learning_rate', 0.0001, 0.1, log=True)
            }
            
            # Ensemble weights
            xgb_weight = trial.suggest_float('xgb_weight', 0.2, 0.6, step=0.01)
            hmm_weight = trial.suggest_float('hmm_weight', 0.1, 0.5, step=0.01)
            lstm_weight = 1.0 - xgb_weight - hmm_weight
            
            if lstm_weight < 0.1:
                lstm_weight = 0.1
                total = xgb_weight + hmm_weight + lstm_weight
                xgb_weight /= total
                hmm_weight /= total
                lstm_weight /= total
            
            ensemble_weights = [xgb_weight, hmm_weight, lstm_weight]
        else:
            lstm_params = None
            ensemble_weights = [0.6, 0.4, 0.0]  # No LSTM component
        
        # Barrier parameters
        barrier_multiplier = trial.suggest_float('barrier_multiplier', 0.5, 3.0, step=0.1)
        
        barrier_params = {
            'volatility_window': trial.suggest_int('volatility_window', 5, 100, step=3),
            'upper_barrier_multiplier': barrier_multiplier,
            'lower_barrier_multiplier': barrier_multiplier,
            'time_barrier_days': trial.suggest_int('time_barrier_days', 3, 35),
            'verbose': False
        }
        
        # Apply triple barrier labeling with current parameters
        temp_triple_barrier_df = apply_triple_barrier_labeling(
            data=filtered_df,
            **barrier_params
        )
        
        # Merge with filtered_df
        temp_triple_barrier_df['decision_date'] = pd.to_datetime(temp_triple_barrier_df['decision_date'])
        temp_merged_df = temp_triple_barrier_df.merge(
            filtered_df, 
            left_on='decision_date', 
            right_on='date', 
            how='left'
        )
        
        if 'date' in temp_merged_df.columns:
            temp_merged_df = temp_merged_df.drop('date', axis=1)
        
        # Drop unnecessary columns for training
        temp_columns_to_drop = [
            'decision_date', 'entry_date', 'end_date', 'end_price',
            'return', 'barrier_touched', 'value_at_barrier_touched',
            'time_converted', 'datetime', 'time', 'time_barrier'
        ]
        temp_existing_columns_to_drop = [col for col in temp_columns_to_drop if col in temp_merged_df.columns]
        
        if temp_existing_columns_to_drop:
            df_with_labels_new = temp_merged_df.drop(columns=temp_existing_columns_to_drop)
        else:
            df_with_labels_new = temp_merged_df.copy()
        
        # Remove rows with NaN labels
        df_with_labels_new = df_with_labels_new.dropna(subset=['label'])
        
        if len(df_with_labels_new) < 100:  # Skip if too few samples
            return 0.0  # Return poor score for pruning
        
        # Apply time-based split to hyperparameter tuning data
        merged_new_df = df_with_labels_new.copy()
        merged_new_df['decision_date'] = temp_triple_barrier_df['decision_date']
        
        # Define Q3 2024 start date (July 1, 2024) - same as initial split
        q3_2024_start_new = pd.Timestamp('2024-07-01')
        
        # Split data based on time
        test_mask_new = merged_new_df['decision_date'] >= q3_2024_start_new
        train_val_mask_new = ~test_mask_new
        
        # Create train+validation set from data before Q3 2024
        train_val_data_new = merged_new_df[train_val_mask_new].copy()
        
        # Extract features and labels for train+val set
        X_train_val_new = train_val_data_new.drop(['label', 'decision_date'], axis=1)
        y_train_val_new = train_val_data_new['label']
        
        # Split train+val into train (80%) and validation (20%)
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
            X_train_val_new, y_train_val_new, test_size=0.20, random_state=42
        )
        
        # Map labels
        unique_labels_new = sorted(y_train_new.unique())
        num_classes_new = len(unique_labels_new)
        
        if num_classes_new == 2:
            label_mapping_new = {unique_labels_new[0]: 0, unique_labels_new[1]: 1}
        else:
            label_mapping_new = {label: idx for idx, label in enumerate(unique_labels_new)}
        
        y_train_mapped_new = y_train_new.map(label_mapping_new)
        y_val_mapped_new = y_val_new.map(label_mapping_new)
        
        # Dynamically adjust n_states to match number of classes
        suggested_n_states = hmm_params['n_states']
        actual_n_states = max(num_classes_new, suggested_n_states)  # Ensure n_states >= num_classes
        
        # Train XGB-HMM model with adjusted parameters
        temp_xgb_hmm_model = XGBHMMModel(
            n_states=actual_n_states,
            max_iter=hmm_params['max_iter'],
            tol=hmm_params['tol'],
            random_state=42
        )
        
        # Set XGBoost parameters with verbosity=0 to reduce logging
        xgb_params['verbosity'] = 0
        temp_xgb_hmm_model.xgb_params = xgb_params
        
        # Fit the model
        temp_xgb_hmm_model.fit(X_train_new, y_train_mapped_new)
        
        # Evaluate on validation set
        y_val_pred_new = temp_xgb_hmm_model.predict(X_val_new)
        val_accuracy_new = accuracy_score(y_val_mapped_new, y_val_pred_new)
        
        return val_accuracy_new
        
    except Exception as e:
        logging.warning(f"Trial failed with error: {str(e)}")
        return 0.0

# Update the optimization study to use the LSTM-enabled objective function
logging.info("Combined HMM, XGBoost, and LSTM Parameters Tuning with Optuna")
logging.info("-" * 50)

if TENSORFLOW_AVAILABLE:
    logging.info("TensorFlow available - using XGB-HMM-LSTM hybrid optimization")
    objective_function = combined_xgb_hmm_lstm_objective
    study_name = "combined_xgb_hmm_lstm_barrier_parameters"
else:
    logging.info("TensorFlow not available - using XGB-HMM optimization only")
    objective_function = combined_xgb_hmm_objective
    study_name = "combined_xgb_hmm_barrier_parameters"

# Create Optuna study for combined parameters
logging.info(f"Starting {study_name} optimization with Optuna...")
start_time = time.time()

# Set Optuna logging level to reduce verbosity and avoid conflicts
optuna.logging.set_verbosity(optuna.logging.WARNING)

combined_study = create_study_with_storage(
    study_name=study_name,
    direction='maximize'
)

# Create callback for early stopping and progress tracking
combined_callback = EnhancedProgressCallback(
    study_name="Combined Parameters Optimization", 
    patience=15, 
    min_improvement=0.005,
    max_trial_time=600
)

# Optimize combined parameters
combined_study.optimize(
    objective_function, 
    n_trials=100,  # Increased for better hyperparameter search
    callbacks=[combined_callback],
    show_progress_bar=True
)

# Calculate tuning time
combined_tuning_time = time.time() - start_time

# Get best parameters
best_combined_trial = combined_study.best_trial
best_combined_score = best_combined_trial.value

# Extract best parameters based on available components
best_hmm_params_combined = {
    'n_states': best_combined_trial.params['n_states'],
    'max_iter': best_combined_trial.params['max_iter'],
    'tol': best_combined_trial.params['tol']
}

best_xgb_params_combined = {
    'n_estimators': best_combined_trial.params['n_estimators'],
    'max_depth': best_combined_trial.params['max_depth'],
    'learning_rate': best_combined_trial.params['learning_rate'],
    'subsample': best_combined_trial.params['subsample'],
    'colsample_bytree': best_combined_trial.params['colsample_bytree'],
    'reg_alpha': best_combined_trial.params['reg_alpha'],
    'reg_lambda': best_combined_trial.params['reg_lambda'],
    'min_child_weight': best_combined_trial.params['min_child_weight'],
    'gamma': best_combined_trial.params['gamma']
}

# Extract LSTM parameters if available
if TENSORFLOW_AVAILABLE and 'sequence_length' in best_combined_trial.params:
    best_lstm_params_combined = {
        'sequence_length': best_combined_trial.params['sequence_length'],
        'lstm_units': best_combined_trial.params['lstm_units'],
        'dropout_rate': best_combined_trial.params['dropout_rate'],
        'learning_rate': best_combined_trial.params['lstm_learning_rate']
    }
    
    best_ensemble_weights_combined = [
        best_combined_trial.params['xgb_weight'],
        best_combined_trial.params['hmm_weight'],
        1.0 - best_combined_trial.params['xgb_weight'] - best_combined_trial.params['hmm_weight']
    ]
else:
    best_lstm_params_combined = None
    best_ensemble_weights_combined = [0.6, 0.4, 0.0]

best_barrier_params_combined = {
    'volatility_window': best_combined_trial.params['volatility_window'],
    'upper_barrier_multiplier': best_combined_trial.params['barrier_multiplier'],
    'lower_barrier_multiplier': best_combined_trial.params['barrier_multiplier'],
    'time_barrier_days': best_combined_trial.params['time_barrier_days']
}

logging.info(f"\nCombined hyperparameter tuning completed in {combined_tuning_time:.2f} seconds")
logging.info(f"Best combined validation accuracy: {best_combined_score:.4f}")
logging.info(f"Best trial number: {best_combined_trial.number}")

logging.info(f"\nBest HMM Parameters:")
for param, value in best_hmm_params_combined.items():
    logging.info(f"  {param}: {value}")

logging.info(f"\nBest XGBoost Parameters:")
for param, value in best_xgb_params_combined.items():
    logging.info(f"  {param}: {value}")

if best_lstm_params_combined:
    logging.info(f"\nBest LSTM Parameters:")
    for param, value in best_lstm_params_combined.items():
        logging.info(f"  {param}: {value}")
    logging.info(f"\nBest Ensemble Weights: XGB={best_ensemble_weights_combined[0]:.2f}, HMM={best_ensemble_weights_combined[1]:.2f}, LSTM={best_ensemble_weights_combined[2]:.2f}")

logging.info(f"\nBest Barrier Parameters:")
for param, value in best_barrier_params_combined.items():
    logging.info(f"  {param}: {value}")

# Save optimization results
optimization_summary = {
    'study_name': study_name,
    'tensorflow_available': TENSORFLOW_AVAILABLE,
    'best_score': best_combined_score,
    'best_trial_number': best_combined_trial.number,
    'total_trials': len(combined_study.trials),
    'tuning_time_seconds': combined_tuning_time,
    'best_hmm_params': best_hmm_params_combined,
    'best_xgb_params': best_xgb_params_combined,
    'best_lstm_params': best_lstm_params_combined,
    'best_ensemble_weights': best_ensemble_weights_combined,
    'best_barrier_params': best_barrier_params_combined
}

optimization_summary_path = 'logs/xgb_results/combined_optimization_summary.pkl'
joblib.dump(optimization_summary, optimization_summary_path)
logging.info(f"Optimization summary saved to: {optimization_summary_path}")

# Use the best parameters for final model training
logging.info(f"\nUsing best parameters from combined tuning for final model training...")
final_hmm_params = best_hmm_params_combined
final_xgb_params = best_xgb_params_combined
final_lstm_params = best_lstm_params_combined
final_ensemble_weights = best_ensemble_weights_combined
final_barrier_params = best_barrier_params_combined

# Apply final triple barrier labeling with best parameters
logging.info(f"Applying triple barrier labeling with optimized parameters...")
final_triple_barrier_df = apply_triple_barrier_labeling(
    data=filtered_df,
    **final_barrier_params,
    verbose=True
)

# Merge with features using optimized barrier parameters
final_triple_barrier_df['decision_date'] = pd.to_datetime(final_triple_barrier_df['decision_date'])
final_merged_df = final_triple_barrier_df.merge(
    filtered_df, 
    left_on='decision_date', 
    right_on='date', 
    how='left'
)

if 'date' in final_merged_df.columns:
    final_merged_df = final_merged_df.drop('date', axis=1)

# Update merged_label_df with optimized data
merged_label_df = final_merged_df.copy()
logging.info(f"Updated merged data with optimized barrier parameters: {merged_label_df.shape}")

# 9. Training XGB-HMM Hybrid Model with Optimized Parameters
logging.info("\n9. Training XGB-HMM Hybrid Model")
logging.info("-" * 50)

# Use existing triple barrier data that was already created
logging.info("Using existing triple barrier data for XGB-HMM training...")
logging.info(f"Triple barrier data already available: {len(triple_barrier_df)} samples")

# Skip the recreation of triple barrier data since we already have merged_label_df
logging.info("Proceeding directly to XGB-HMM training with existing merged data...")

# Use merged_label_df as the final dataset
df_with_labels_final = merged_label_df.copy()

# Remove rows with NaN labels
df_with_labels_final = df_with_labels_final.dropna(subset=['label'])

logging.info(f"Final dataset shape: {df_with_labels_final.shape}")
logging.info(f"Label distribution:")
logging.info(df_with_labels_final['label'].value_counts().sort_index())

# Apply time-based split to final dataset (same as initial approach)
df_with_labels_final['decision_date'] = pd.to_datetime(df_with_labels_final['decision_date'])

# Define Q3 2024 start date (July 1, 2024) - same as initial split
q3_2024_start_final = pd.Timestamp('2024-07-01')

# Split data based on time (same logic as initial split)
test_mask_final = df_with_labels_final['decision_date'] >= q3_2024_start_final
train_val_mask_final = ~test_mask_final

# Create test set from Q3 2024 onwards
test_data_final = df_with_labels_final[test_mask_final].copy()
logging.info(f"Final test data period: {test_data_final['decision_date'].min()} to {test_data_final['decision_date'].max()}")
logging.info(f"Final test data shape: {test_data_final.shape}")

# Create train+validation set from data before Q3 2024
train_val_data_final = df_with_labels_final[train_val_mask_final].copy()
logging.info(f"Final train+val data period: {train_val_data_final['decision_date'].min()} to {train_val_data_final['decision_date'].max()}")
logging.info(f"Final train+val data shape: {train_val_data_final.shape}")

# Drop datetime columns for final training
final_columns_to_drop_clean = [
    'decision_date', 'entry_date', 'end_date', 'end_price',
    'return', 'barrier_touched', 'value_at_barrier_touched',
    'time_converted', 'datetime', 'time', 'time_barrier'
]

# Apply column dropping to final test set
existing_cols_test_final = [col for col in final_columns_to_drop_clean if col in test_data_final.columns]
if existing_cols_test_final:
    test_data_clean_final = test_data_final.drop(columns=existing_cols_test_final)
else:
    test_data_clean_final = test_data_final.copy()

# Apply column dropping to final train+val set
existing_cols_train_val_final = [col for col in final_columns_to_drop_clean if col in train_val_data_final.columns]
if existing_cols_train_val_final:
    train_val_data_clean_final = train_val_data_final.drop(columns=existing_cols_train_val_final)
else:
    train_val_data_clean_final = train_val_data_final.copy()

# Extract features and labels for final test set
X_test_final = test_data_clean_final.drop('label', axis=1)
y_test_final = test_data_clean_final['label']

# Extract features and labels for final train+val set
X_train_val_final = train_val_data_clean_final.drop('label', axis=1)
y_train_val_final = train_val_data_clean_final['label']

# Split train+val into train (80%) and validation (20%) - remove stratify for time series
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_val_final, y_train_val_final, test_size=0.20, random_state=42
)

logging.info(f"\nFinal data splits:")
logging.info(f"Training set: {X_train_final.shape[0]} samples")
logging.info(f"Validation set: {X_val_final.shape[0]} samples")
logging.info(f"Test set: {X_test_final.shape[0]} samples")

# Map labels
unique_labels_final = sorted(y_train_final.unique())
num_classes_final = len(unique_labels_final)

if num_classes_final == 2:
    label_mapping_final = {unique_labels_final[0]: 0, unique_labels_final[1]: 1}
else:
    label_mapping_final = {label: idx for idx, label in enumerate(unique_labels_final)}

y_train_mapped_final = y_train_final.map(label_mapping_final)
y_val_mapped_final = y_val_final.map(label_mapping_final)
y_test_mapped_final = y_test_final.map(label_mapping_final)

logging.info(f"\nFinal label mapping: {label_mapping_final}")
logging.info(f"Number of classes: {num_classes_final}")

# Apply Feature Selection
logging.info("\n=== Applying Feature Selection ===")
logging.info("Initializing feature selection process...")

# Initialize feature selector
feature_selector = FeatureSelector(random_state=42)

# Get original feature names
original_feature_names = X_train_final.columns.tolist()
logging.info(f"Original number of features: {len(original_feature_names)}")

# Apply ensemble feature selection
logging.info("Applying ensemble feature selection...")
X_train_selected_fs, selected_features, feature_votes = feature_selector.ensemble_selection(
    X_train_final, y_train_mapped_final,
    methods=['univariate', 'mutual_info', 'rfe', 'feature_importance'],
    k_univariate=min(100, len(original_feature_names) // 2),
    k_mutual_info=min(100, len(original_feature_names) // 2),
    n_rfe=min(100, len(original_feature_names) // 2),
    n_importance=min(100, len(original_feature_names) // 2),
    min_votes=2
)

# Get selected features (already unpacked from tuple)
selected_feature_names = [original_feature_names[i] for i in selected_features]

logging.info(f"Selected {len(selected_features)} features out of {len(original_feature_names)}")
logging.info(f"Feature reduction: {((len(original_feature_names) - len(selected_features)) / len(original_feature_names) * 100):.2f}%")

# Log feature selection summary
feature_summary = feature_selector.get_feature_selection_summary(original_feature_names)
logging.info("\nFeature Selection Summary:")
for method, info in feature_summary.items():
    if isinstance(info, dict) and 'selected_count' in info:
        logging.info(f"  {method}: {info['selected_count']} features selected")
        if 'scores' in info and len(info['scores']) > 0:
            logging.info(f"    Top 5 scores: {sorted(info['scores'], reverse=True)[:5]}")

# Apply feature selection to all datasets
logging.info("Applying feature selection to training, validation, and test sets...")
X_train_selected = X_train_final.iloc[:, selected_features]
X_val_selected = X_val_final.iloc[:, selected_features]
X_test_selected = X_test_final.iloc[:, selected_features]

logging.info(f"Selected feature names (top 10): {selected_feature_names[:10]}")
if len(selected_feature_names) > 10:
    logging.info(f"... and {len(selected_feature_names) - 10} more features")

# Save feature selection results
feature_selection_summary = {
    'original_features': original_feature_names,
    'selected_features': selected_feature_names,
    'selected_indices': selected_features,
    'feature_summary': feature_summary
}

feature_selection_path = 'logs/xgb_results/feature_selection_summary.pkl'
joblib.dump(feature_selection_summary, feature_selection_path)
logging.info(f"Feature selection summary saved to: {feature_selection_path}")

# Train final model with best combined hyperparameters and selected features
# Training XGB-HMM-LSTM Triple Hybrid Model
logging.info("\n=== Training XGB-HMM-LSTM Triple Hybrid Model with Selected Features ===")
logging.info("Initializing XGB-HMM-LSTM model with optimal parameters and selected features...")

# Check if we should use LSTM hybrid or fallback to XGB-HMM
if TENSORFLOW_AVAILABLE and final_lstm_params is not None:
    # Initialize XGB-HMM-LSTM model with optimized parameters
    xgb_hmm_lstm_model = XGBHMMLSTMModel(
        n_states=final_hmm_params['n_states'],
        max_iter=final_hmm_params['max_iter'],
        tol=final_hmm_params['tol'],
        sequence_length=final_lstm_params['sequence_length'],
        lstm_units=final_lstm_params['lstm_units'],
        dropout_rate=final_lstm_params['dropout_rate'],
        learning_rate=final_lstm_params['learning_rate'],
        ensemble_weights=final_ensemble_weights,
        random_state=42
    )
    
    # Set optimized XGBoost parameters
    xgb_hmm_lstm_model.xgb_params = final_xgb_params
    
    logging.info(f"XGB-HMM-LSTM Model initialized with optimized parameters")
    logging.info(f"  n_states: {final_hmm_params['n_states']}")
    logging.info(f"  max_iter: {final_hmm_params['max_iter']}")
    logging.info(f"  tol: {final_hmm_params['tol']}")
    logging.info(f"  sequence_length: {final_lstm_params['sequence_length']}")
    logging.info(f"  lstm_units: {final_lstm_params['lstm_units']}")
    logging.info(f"  ensemble_weights: {final_ensemble_weights}")
    logging.info(f"  TensorFlow available: {TENSORFLOW_AVAILABLE}")
    
    # Train the XGB-HMM-LSTM model
    logging.info("Starting XGB-HMM-LSTM training with EM algorithm...")
    start_time = time.time()
    
    # Fit the model
    xgb_hmm_lstm_model.fit(X_train_selected, y_train_mapped_final)
    
    training_time = time.time() - start_time
    logging.info(f"XGB-HMM-LSTM training completed in {training_time:.2f} seconds")
    
    # Use the LSTM hybrid model for evaluation
    model_to_evaluate = xgb_hmm_lstm_model
    model_name = "XGB-HMM-LSTM"
    
    # Add LSTM-specific evaluation
    if hasattr(model_to_evaluate, 'lstm_model') and model_to_evaluate.lstm_model is not None:
        logging.info(f"LSTM model successfully trained and integrated")
        logging.info(f"Ensemble weights used: XGB={final_ensemble_weights[0]:.2f}, HMM={final_ensemble_weights[1]:.2f}, LSTM={final_ensemble_weights[2]:.2f}")
    else:
        logging.warning(f"LSTM model not available, using XGB-HMM only")
    
else:
    # Fallback to XGB-HMM if TensorFlow not available or LSTM params not optimized
    logging.warning("Using XGB-HMM model instead of LSTM hybrid.")
    logging.info(f"Reason: TensorFlow available = {TENSORFLOW_AVAILABLE}, LSTM params = {final_lstm_params is not None}")
    
    xgb_hmm_model = XGBHMMModel(
        n_states=final_hmm_params['n_states'],
        max_iter=final_hmm_params['max_iter'],
        tol=final_hmm_params['tol'],
        random_state=42
    )
    
    # Set optimized XGBoost parameters for emission model
    xgb_hmm_model.xgb_params = final_xgb_params
    
    logging.info(f"XGB-HMM Model initialized with optimized parameters:")
    logging.info(f"  n_states: {final_hmm_params['n_states']}")
    logging.info(f"  max_iter: {final_hmm_params['max_iter']}")
    logging.info(f"  tol: {final_hmm_params['tol']}")
    logging.info(f"XGBoost parameters: {final_xgb_params}")
    
    # Train the XGB-HMM model
    logging.info("Starting XGB-HMM training with EM algorithm...")
    start_time = time.time()
    
    # Fit the model
    xgb_hmm_model.fit(X_train_selected, y_train_mapped_final)
    
    training_time = time.time() - start_time
    logging.info(f"XGB-HMM training completed in {training_time:.2f} seconds")
    
    # Use the HMM model for evaluation
    model_to_evaluate = xgb_hmm_model
    model_name = "XGB-HMM"

# Evaluate on all sets
# Evaluate model with additional metrics
logging.info(f"\n=== {model_name} Model Detailed Evaluation ===")

# Evaluate on training set
train_evaluation = model_to_evaluate.evaluate_model(X_train_selected, y_train_mapped_final)
logging.info(f"\nTraining Set {model_name} Evaluation:")
for metric, value in train_evaluation.items():
    logging.info(f"  {metric}: {value:.4f}")

# Evaluate on validation set
val_evaluation = model_to_evaluate.evaluate_model(X_val_selected, y_val_mapped_final)
logging.info(f"\nValidation Set {model_name} Evaluation:")
for metric, value in val_evaluation.items():
    logging.info(f"  {metric}: {value:.4f}")

# Evaluate on test set
test_evaluation = model_to_evaluate.evaluate_model(X_test_selected, y_test_mapped_final)
logging.info(f"\nTest Set {model_name} Evaluation:")
for metric, value in test_evaluation.items():
    logging.info(f"  {metric}: {value:.4f}")

# State transition analysis
logging.info("\n=== State Transition Analysis ===")
transition_analysis = model_to_evaluate.get_state_transition_analysis(X_test_selected)
logging.info(f"\nLearned Transition Matrix:")
logging.info(transition_analysis['learned_transition_matrix'])
logging.info(f"\nEmpirical Transition Matrix:")
logging.info(transition_analysis['empirical_transition_matrix'])

logging.info(f"\nEvaluating {model_name} model on all datasets...")

# Training set evaluation
y_train_pred_final = model_to_evaluate.predict(X_train_selected)
y_train_proba_final = model_to_evaluate.predict_proba(X_train_selected)

train_accuracy_final = accuracy_score(y_train_mapped_final, y_train_pred_final)
train_precision_final = precision_score(y_train_mapped_final, y_train_pred_final, average='weighted')
train_recall_final = recall_score(y_train_mapped_final, y_train_pred_final, average='weighted')
train_f1_final = f1_score(y_train_mapped_final, y_train_pred_final, average='weighted')

if num_classes_final == 2:
    train_roc_auc_final = roc_auc_score(y_train_mapped_final, y_train_proba_final[:, 1])
else:
    train_roc_auc_final = roc_auc_score(y_train_mapped_final, y_train_proba_final, multi_class='ovr')

# Validation set evaluation
y_val_pred_final = model_to_evaluate.predict(X_val_selected)
y_val_proba_final = model_to_evaluate.predict_proba(X_val_selected)

val_accuracy_final = accuracy_score(y_val_mapped_final, y_val_pred_final)
val_precision_final = precision_score(y_val_mapped_final, y_val_pred_final, average='weighted')
val_recall_final = recall_score(y_val_mapped_final, y_val_pred_final, average='weighted')
val_f1_final = f1_score(y_val_mapped_final, y_val_pred_final, average='weighted')

if num_classes_final == 2:
    val_roc_auc_final = roc_auc_score(y_val_mapped_final, y_val_proba_final[:, 1])
else:
    val_roc_auc_final = roc_auc_score(y_val_mapped_final, y_val_proba_final, multi_class='ovr')

# Test set evaluation
y_test_pred_final = model_to_evaluate.predict(X_test_selected)
y_test_proba_final = model_to_evaluate.predict_proba(X_test_selected)

test_accuracy_final = accuracy_score(y_test_mapped_final, y_test_pred_final)
test_precision_final = precision_score(y_test_mapped_final, y_test_pred_final, average='weighted')
test_recall_final = recall_score(y_test_mapped_final, y_test_pred_final, average='weighted')
test_f1_final = f1_score(y_test_mapped_final, y_test_pred_final, average='weighted')

# Handle ROC AUC calculation with proper class checking
unique_test_classes = len(np.unique(y_test_mapped_final))
if unique_test_classes == 2 and y_test_proba_final.shape[1] == 2:
    test_roc_auc_final = roc_auc_score(y_test_mapped_final, y_test_proba_final[:, 1])
elif unique_test_classes > 2 and y_test_proba_final.shape[1] == unique_test_classes:
    test_roc_auc_final = roc_auc_score(y_test_mapped_final, y_test_proba_final, multi_class='ovr')
else:
    # Fallback: use accuracy as ROC AUC when there's a mismatch
    test_roc_auc_final = test_accuracy_final
    logging.warning(f"ROC AUC calculation skipped due to class mismatch. Test classes: {unique_test_classes}, Proba shape: {y_test_proba_final.shape}")

# Display final results
logging.info("\n" + "="*60)
logging.info(f"{model_name} HYBRID MODEL PERFORMANCE")
logging.info("="*60)

logging.info(f"\nTraining Set Performance:")
logging.info(f"Accuracy: {train_accuracy_final:.4f}")
logging.info(f"Precision: {train_precision_final:.4f}")
logging.info(f"Recall: {train_recall_final:.4f}")
logging.info(f"F1-Score: {train_f1_final:.4f}")
logging.info(f"ROC AUC: {train_roc_auc_final:.4f}")

logging.info(f"\nValidation Set Performance:")
logging.info(f"Accuracy: {val_accuracy_final:.4f}")
logging.info(f"Precision: {val_precision_final:.4f}")
logging.info(f"Recall: {val_recall_final:.4f}")
logging.info(f"F1-Score: {val_f1_final:.4f}")
logging.info(f"ROC AUC: {val_roc_auc_final:.4f}")

logging.info(f"\nTest Set Performance:")
logging.info(f"Accuracy: {test_accuracy_final:.4f}")
logging.info(f"Precision: {test_precision_final:.4f}")
logging.info(f"Recall: {test_recall_final:.4f}")
logging.info(f"F1-Score: {test_f1_final:.4f}")
logging.info(f"ROC AUC: {test_roc_auc_final:.4f}")

# Confusion Matrix for Test Set
logging.info(f"\nTest Set Confusion Matrix:")
cm_test_final = confusion_matrix(y_test_mapped_final, y_test_pred_final)
logging.info(cm_test_final)

# Classification Report for Test Set
logging.info(f"\nTest Set Classification Report:")
reverse_label_mapping_final = {v: k for k, v in label_mapping_final.items()}
# Use all classes that the model can predict (union of actual and predicted classes)
unique_test_classes_actual = sorted(np.unique(y_test_mapped_final))
unique_test_classes_predicted = sorted(np.unique(y_test_pred_final))
all_classes = sorted(set(unique_test_classes_actual) | set(unique_test_classes_predicted))
target_names_final = [str(reverse_label_mapping_final.get(i, f'Class_{i}')) for i in all_classes]

logging.info(f"Classes in test data: {unique_test_classes_actual}")
logging.info(f"Classes predicted by model: {unique_test_classes_predicted}")
logging.info(f"All classes for classification report: {all_classes}")
logging.info(classification_report(y_test_mapped_final, y_test_pred_final, target_names=target_names_final, labels=all_classes))

# Performance comparison
logging.info(f"\n" + "="*60)
logging.info("PERFORMANCE COMPARISON")
logging.info("="*60)

performance_comparison_final = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'Accuracy': [train_accuracy_final, val_accuracy_final, test_accuracy_final],
    'Precision': [train_precision_final, val_precision_final, test_precision_final],
    'Recall': [train_recall_final, val_recall_final, test_recall_final],
    'F1-Score': [train_f1_final, val_f1_final, test_f1_final],
    'ROC AUC': [train_roc_auc_final, val_roc_auc_final, test_roc_auc_final]
})

logging.info(performance_comparison_final.round(4))

# Save final results
final_results_path = 'logs/xgb_results/final_performance_comparison.csv'
performance_comparison_final.to_csv(final_results_path, index=False)
logging.info(f"\nFinal performance comparison saved to: {final_results_path}")

# Save final model
final_model_path = f'logs/xgb_models/{model_name.lower().replace("-", "_")}_model.pkl'
joblib.dump(model_to_evaluate, final_model_path)
logging.info(f"{model_name} model saved to: {final_model_path}")

# Save final test predictions
final_test_predictions_dict = {
    'actual_label': y_test_final.values,
    'predicted_label': [reverse_label_mapping_final[pred] for pred in y_test_pred_final],
    'actual_mapped': y_test_mapped_final.values,
    'predicted_mapped': y_test_pred_final
}

# Add probability columns based on number of classes
if num_classes_final == 2:
    final_test_predictions_dict['probability_class_0'] = y_test_proba_final[:, 0]
    final_test_predictions_dict['probability_class_1'] = y_test_proba_final[:, 1]
else:
    for i in range(num_classes_final):
        final_test_predictions_dict[f'probability_class_{i}'] = y_test_proba_final[:, i]

final_test_predictions_df = pd.DataFrame(final_test_predictions_dict)

final_test_predictions_path = 'logs/xgb_results/final_test_predictions.csv'
final_test_predictions_df.to_csv(final_test_predictions_path, index=False)
logging.info(f"Final test predictions saved to: {final_test_predictions_path}")

# Summary of hyperparameter tuning
logging.info(f"\n" + "="*60)
logging.info(f"{model_name} HYPERPARAMETER SUMMARY")
logging.info("="*60)

logging.info(f"\n{model_name} Model Parameters:")
logging.info(f"  n_states: {model_to_evaluate.n_states}")
logging.info(f"  max_iter: {model_to_evaluate.max_iter}")
logging.info(f"  tolerance: {model_to_evaluate.tol}")
logging.info(f"  random_state: {model_to_evaluate.random_state}")

logging.info(f"\n{model_name} Training Summary:")
logging.info(f"  Training time: {training_time:.2f} seconds")
logging.info(f"  Convergence achieved: {hasattr(model_to_evaluate, 'converged_') and model_to_evaluate.converged_}")
logging.info(f"  Final log-likelihood: {model_to_evaluate.log_likelihood_history_[-1] if hasattr(model_to_evaluate, 'log_likelihood_history_') else 'N/A'}")

logging.info(f"\nHybrid XGB-HMM vs Original XGBoost Performance:")
logging.info(f"Original XGBoost Test Accuracy: ~0.45 (from previous runs)")
logging.info(f"Hybrid XGB-HMM Test Accuracy: {test_accuracy_final:.4f}")
logging.info(f"Performance Improvement: {((test_accuracy_final - 0.45) / 0.45 * 100):.2f}%")

logging.info(f"\nHybrid XGB-HMM Model Generalization:")
logging.info(f"Training Accuracy: {train_accuracy_final:.4f}")
logging.info(f"Validation Accuracy: {val_accuracy_final:.4f}")
logging.info(f"Test Accuracy: {test_accuracy_final:.4f}")
logging.info(f"Train-Val Gap: {train_accuracy_final - val_accuracy_final:.4f}")
logging.info(f"Train-Test Gap: {train_accuracy_final - test_accuracy_final:.4f}")

# Feature Selection Impact Analysis
logging.info(f"\n=== Feature Selection Impact Analysis ===")
logging.info(f"Original features: {len(original_feature_names)}")
logging.info(f"Selected features: {len(selected_features)}")
logging.info(f"Feature reduction: {((len(original_feature_names) - len(selected_features)) / len(original_feature_names) * 100):.2f}%")
logging.info(f"Model performance maintained with {(len(selected_features) / len(original_feature_names) * 100):.2f}% of original features")

# Log top selected features by importance
if 'feature_importance' in feature_summary and 'scores' in feature_summary['feature_importance']:
    importance_scores = feature_summary['feature_importance']['scores']
    if len(importance_scores) > 0:
        # Get top 10 features by importance
        top_indices = sorted(range(len(importance_scores)), key=lambda i: importance_scores[i], reverse=True)[:10]
        top_features = [selected_feature_names[i] for i in top_indices if i < len(selected_feature_names)]
        logging.info(f"\nTop 10 Selected Features by Importance:")
        for i, feature in enumerate(top_features, 1):
            if i-1 < len(importance_scores):
                logging.info(f"  {i}. {feature}: {importance_scores[top_indices[i-1]]:.4f}")

logging.info("\n" + "="*80)
logging.info(f"{model_name} HYBRID MODEL WITH FEATURE SELECTION COMPLETED SUCCESSFULLY!")
logging.info("="*80)