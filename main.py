import os
import logging
import click
import pandas as pd
from datetime import datetime

from transform import FeatureTransformation
from predict import AnomalyPredictor
from constants import ISOLATION_FOREST_EXPECTED_COLUMNS


# @click.command()
# @click.argument("input_file", type=click.Path(exists=True))
def main(input_file):
    """
    Transform, predict, and save anomaly results for the given input file (.tsv).

    Assumes that the input file is a TSV file without header, however contains following
    columns:
    - datetime
    - region
    - browser
    - device
    - url_params

    Output is two files:
    - a TSV file with the following columns:
        - is_anomaly
        - anomaly_score
        Index is preserved from the input file.
    - a small txt file with small report on the results.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("main")

    # Prep for output
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"output_{timestamp}"
    output_file_name = os.path.join(output_dir, f"{output_name}.tsv")
    output_report_name = os.path.join(output_dir, f"{output_name}_report.txt")

    # Load input
    logger.info(f"Loading input file: {input_file}")
    df = pd.read_csv(
        input_file,
        sep="\t",
        index_col=False,
        header=None,
        names=["datetime", "region", "browser", "device", "url_params"],
    )

    # Transform
    logger.info("Setting up feature transformation...")
    transformer = FeatureTransformation()
    logger.info("Running feature transformation...")
    X = transformer.transform(df)

    # Predict
    logger.info("Setting up anomaly prediction...")
    predictor = AnomalyPredictor()
    logger.info("Running anomaly prediction...")
    results = predictor.predict(X)
    results_df = results["results_df"]
    results_report = results["report"]

    # Merge results with original input columns for output
    output_df = pd.concat(
        [df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1
    )

    # Save output
    output_df.to_csv(output_file_name, sep="\t", index=False)
    logger.info(f"Saved output to {output_file_name}")

    # Save report
    with open(output_report_name, "w") as f:
        f.write(results_report)
    logger.info(f"Saved report to {output_report_name}")


if __name__ == "__main__":
    main("data/bot-hunter-dataset.tsv")
