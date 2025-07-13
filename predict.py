import logging
import pickle
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import IsolationForest

from constants import ANOMALY_SCORE_CUTOFF


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AnomalyPredictor:
    """
    Simple anomaly detection predictor that loads a trained model and makes predictions.
    Uses ANOMALY_SCORE_CUTOFF from constants.py for threshold-based classification.
    """

    def __init__(
        self,
        model_path: str = "models/anomaly_detection_model.pkl",
        cutoff: float = ANOMALY_SCORE_CUTOFF,
    ):
        """
        Initialize the anomaly predictor.

        Args:
            model_path: Path to the trained model pickle file
        """
        self.logger = logging.getLogger(__name__)

        try:
            self.model = pickle.load(open(model_path, "rb"))
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

        self.cutoff = cutoff
        self.logger.info(f"AnomalyPredictor initialized with cutoff: {self.cutoff}")

    def _load_model(self, model_path: str) -> IsolationForest:
        """Load a trained model from pickle file."""
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            self.logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make anomaly predictions on pre-processed data.

        Args:
            X: Pre-processed DataFrame (output from transform.py)

        Returns:
            Dictionary containing predictions and scores
        """
        self.logger.info(f"Making predictions on {len(X)} samples")
        anomaly_scores_decision = self.model.decision_function(X)
        is_anomaly = anomaly_scores_decision < self.cutoff

        results_df = X.copy()
        results_df["anomaly_score"] = anomaly_scores_decision
        results_df["is_anomaly"] = is_anomaly
        results_df["anomaly_label"] = results_df["is_anomaly"].replace(
            {True: "Anomaly", False: "Normal"}
        )

        stats = self._calculate_statistics(results_df)

        self.logger.info(
            f"Prediction completed. Anomalies detected: {stats['n_anomalies']}"
        )

        return {
            "results_df": results_df,
            "anomaly_scores_decision": anomaly_scores_decision,
            "is_anomaly": is_anomaly,
            "statistics": stats,
        }

    def _calculate_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate prediction statistics."""
        stats = {}

        # Basic statistics
        stats["n_total"] = len(results_df)
        stats["n_anomalies"] = results_df["is_anomaly"].sum()
        stats["anomaly_percentage"] = (stats["n_anomalies"] / stats["n_total"]) * 100

        # Score statistics
        stats["mean_score"] = results_df["anomaly_score"].mean()
        stats["std_score"] = results_df["anomaly_score"].std()
        stats["min_score"] = results_df["anomaly_score"].min()
        stats["max_score"] = results_df["anomaly_score"].max()

        # Cutoff information
        stats["cutoff_used"] = self.cutoff

        return stats
