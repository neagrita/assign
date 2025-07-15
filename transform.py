import logging
import pickle
from typing import Optional
from urllib.parse import unquote

import fasttext
import numpy as np
import pandas as pd

from constants import (
    COUNTRY_CODE_FREQ,
    COUNTRY_CODES,
    LANG_LOCALE_FREQ,
    TTC_BINS,
    TTC_LABELS_INTS,
)
from helpers import entropy_scipy, get_url_parts, parse_url_params_simple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FeatureTransformation:
    """
    Feature engineering pipeline that replicates all transformations
    from the feature engineering notebook.
    """

    # column mappings
    CATEGORICAL_COLUMNS = ["browser", "device", "region"]
    COLUMNS_TO_DROP = ["st", "n", "f", "adx_name", "atbexp", "va", "atbva"]
    SPARSE_COLUMNS_TO_ENCODE = ["r", "kp", "sld"]
    SPARSE_COLUMNS_TO_COMBINE = ["lsexp1", "om", "adx", "bkl", "atb"]

    # replace fake nulls to np.nan
    FAKE_NULLS_TO_NAN = ["null", "none"]

    # column type transformations
    COLUMNS_TO_INT = ["nt"]

    # input columns; expected all as str
    INPUT_COLUMNS = ["datetime", "browser", "device", "region", "url_params"]

    def __init__(
        self,
        expected_output_columns: list[str],
        fasttext_model_path: str = "models/cc.en.300.bin",
        svd_model_path: str = "models/svd_model.pkl",
        log_level: str = "INFO",
    ):
        """
        Initialize the pipeline.

        Args:
            fasttext_model_path: Path to FastText model file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """

        # init logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # init fasttext model for embeddings
        try:
            self.fasttext_model = fasttext.load_model(fasttext_model_path)
        except Exception as e:
            logging.error(
                f"Error loading FastText model: {e}\n\n"
                "Transformations require the English FastText model (cc.en.300.bin). "
                "Please download it from https://fasttext.cc/docs/en/crawl-vectors.html, "
                "and place the file in the 'models' folder as 'cc.en.300.bin'."
            )
            raise e

        # init svd model for sparse features
        try:
            self.svd_model = pickle.load(open(svd_model_path, "rb"))
        except Exception as e:
            logging.error(f"Error loading SVD model: {e}")
            raise e

        # init expected output columns
        self.expected_output_columns = expected_output_columns

        self.logger.info("FeatureTransformation initialized")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input dataframe.
        """
        self.logger.info(
            f"Starting transformation pipeline with {len(df)} rows "
            f"and {len(df.columns)} columns"
        )

        df_transformed = self._clean_data(df)
        self.logger.info(
            f"After cleaning: {len(df_transformed)} rows and "
            f"{len(df_transformed.columns)} columns"
        )

        df_transformed = self._encode_categorical_features(
            df_transformed, categorical_columns=self.CATEGORICAL_COLUMNS
        )
        self.logger.info(
            f"After categorical encoding: {len(df_transformed)} rows and "
            f"{len(df_transformed.columns)} columns"
        )

        df_transformed = self._drop_constant_columns(df_transformed)
        self.logger.info(
            f"After dropping constant columns: {len(df_transformed)} rows and "
            f"{len(df_transformed.columns)} columns"
        )

        df_transformed = self._encode_categorical_features(
            df_transformed,
            categorical_columns=self.SPARSE_COLUMNS_TO_ENCODE,
            drop_first=False,
            missing_values=True,
            values_to_missing=self.FAKE_NULLS_TO_NAN,
        )
        self.logger.info(
            f"After sparse encoding: {len(df_transformed)} rows and "
            f"{len(df_transformed.columns)} columns"
        )

        df_transformed = self._combine_sparse_features(df_transformed)
        self.logger.info(
            f"After SVD sparse combination: {len(df_transformed)} rows and "
            f"{len(df_transformed.columns)} columns"
        )

        df_transformed = self._process_ttc(df_transformed)
        df_transformed = self._process_country_code(df_transformed)
        df_transformed = self._process_locale_code(df_transformed)

        df_transformed = self._extract_domain_features(df_transformed)
        self.logger.info(
            f"After domain feature extraction: {len(df_transformed)} rows and "
            f"{len(df_transformed.columns)} columns"
        )

        df_transformed = self._process_query(df_transformed)
        self.logger.info(
            f"After query processing: {len(df_transformed)} rows and "
            f"{len(df_transformed.columns)} columns"
        )

        self.logger.info("Checking if all expected columns are present")
        missing_cols = [
            col
            for col in self.expected_output_columns
            if col not in df_transformed.columns
        ]
        if len(missing_cols) > 0:
            self.logger.warning(f"Missing columns: {missing_cols}")
            self.logger.info("Correcting missing columns")
            df_transformed[missing_cols] = 0

        self.logger.info("Transformation pipeline completed successfully")
        return df_transformed

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Unnest ad click url params and extract hour of day.
        """
        self.logger.debug("Starting data cleaning")

        # Check for missing required columns
        missing_cols = [col for col in self.INPUT_COLUMNS if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing required columns: {missing_cols}")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df.loc[:, "hour"] = df["datetime"].dt.hour
        df.loc[:, "params_dict"] = df["url_params"].apply(parse_url_params_simple)
        params_df = df["params_dict"].apply(pd.Series)
        if params_df.shape[0] != df.shape[0]:
            self.logger.error(
                f"params_df has {params_df.shape[0]} rows, "
                f"but df has {df.shape[0]} rows"
            )
            raise
        params_df.index = df.index
        df = pd.concat([df, params_df], axis=1)

        # direct trasnformations
        df[self.COLUMNS_TO_INT] = df[self.COLUMNS_TO_INT].astype(int)

        # Log any parsing issues
        null_count = df["params_dict"].isna().sum()
        if null_count > 0:
            self.logger.warning(
                f"Found {null_count} rows with unparseable URL parameters"
            )

        return df.drop(columns=["url_params", "datetime", "params_dict"])

    def _encode_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str],
        drop_first: bool = True,
        missing_values: bool = False,
        values_to_missing: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        One-hot encode categorical features.
        If drop_first is True, drop first category to avoid multicollinearity in linear
        models, thus returns len(CATEGORICAL_COLUMNS) less columns.
        If missing_values is True, encode missing values as a separate category.
        If values_to_missing is provided, replace these values with np.nan.
        """
        kwargs = {}
        if drop_first:
            kwargs["drop_first"] = True
        if missing_values:
            kwargs["dummy_na"] = True
        if values_to_missing:
            replacement_dict = {
                col: {val: np.nan for val in values_to_missing}
                for col in categorical_columns
            }
            df = df.replace(replacement_dict)

        dummies = pd.get_dummies(df[categorical_columns], **kwargs).astype(int)
        return pd.concat([df.drop(columns=categorical_columns), dummies], axis=1)

    def _drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that are constant.
        """
        columns_to_drop = [col for col in self.COLUMNS_TO_DROP if col in df.columns]
        return df.drop(columns=columns_to_drop)

    def _combine_sparse_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine sparse features with truncated singular value decomposition.
        """
        self.logger.debug(
            f"Starting SVD on {len(self.SPARSE_COLUMNS_TO_COMBINE)} sparse columns"
        )

        sparse_columns_missing = [
            col for col in self.SPARSE_COLUMNS_TO_COMBINE if col not in df.columns
        ]
        if len(sparse_columns_missing) > 0:
            self.logger.warning(f"Missing sparse columns: {sparse_columns_missing}")
            self.logger.info("Initializing sparse features with missing values")
            df[sparse_columns_missing] = np.nan

        # get sparse features - ensure it's a DataFrame
        sparse_df = df[self.SPARSE_COLUMNS_TO_COMBINE]
        if isinstance(sparse_df, pd.Series):
            sparse_df = sparse_df.to_frame()

        try:
            sparse_dummies = self._encode_categorical_features(
                sparse_df,
                categorical_columns=self.SPARSE_COLUMNS_TO_COMBINE,
                drop_first=False,
                missing_values=True,
                values_to_missing=self.FAKE_NULLS_TO_NAN,
            )

            # if any values are missing, add them here
            add_cols = list(
                set(self.svd_model.feature_names_in_) - set(sparse_dummies.columns)
            )
            if len(add_cols) > 0:
                self.logger.warning(f"Adding {len(add_cols)} missing columns")
                sparse_dummies[add_cols] = 0

            # if any values are new, log that
            new_cols = list(
                set(sparse_dummies.columns) - set(self.svd_model.feature_names_in_)
            )
            if len(new_cols) > 0:
                self.logger.warning(f"New values in columns: {new_cols}")
                sparse_dummies.drop(columns=new_cols, inplace=True)

        except Exception as e:
            self.logger.warning(f"Error getting sparse dummies: {e}")
            self.logger.info("Initializing dummies with missing values")

            add_cols = [col + "_NA" for col in self.SPARSE_COLUMNS_TO_COMBINE]
            sparse_dummies = pd.DataFrame(
                index=df.index, columns=self.svd_model.feature_names_in_, dtype=int
            ).fillna(0)
            sparse_dummies[add_cols] = 1

        self.logger.debug(
            f"Sparse dummy encoding created {sparse_dummies.shape[1]} features"
        )

        svd_features = self.svd_model.transform(
            sparse_dummies[self.svd_model.feature_names_in_]
        )
        svd_df = pd.DataFrame(
            svd_features,
            columns=[f"sparse_{i}" for i in range(self.svd_model.n_components)],  # type: ignore
            index=df.index,
        )

        return pd.concat(
            [df.drop(columns=self.SPARSE_COLUMNS_TO_COMBINE, axis=1), svd_df],
            axis=1,
        )

    def _process_ttc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Log transform time-to-click and bin it.
        """
        df["ttc_log"] = np.log1p(df["ttc"].astype(float))
        df["ttc_bucket"] = pd.cut(
            df["ttc"].astype(float), bins=TTC_BINS, labels=TTC_LABELS_INTS
        ).astype(int)  # type: ignore
        return df.drop(columns=["ttc"])

    def _process_country_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process country code.
        """
        is_country = df["ct"].str.upper().fillna("").isin(COUNTRY_CODES)
        invalid_count = sum(~is_country)
        if invalid_count > 0:
            self.logger.warning(
                f"Found {invalid_count} invalid country codes, replacing with 'unknown'"
            )
            df.loc[~is_country, "ct"] = "unknown"

        df["ct_freq"] = df["ct"].map(COUNTRY_CODE_FREQ).fillna(0)  # type: ignore
        return df.drop(columns=["ct"])

    def _process_locale_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process locale code.
        """

        kl_df = df["kl"].str.split("-", expand=True)
        kl_df.columns = ["locale", "country"]

        # hard-coded fix for data quality
        kl_df.loc[kl_df["country"] == "uk", "country"] = "gb"
        kl_df.loc[kl_df["locale"] == "tzh", "locale"] = "zh"

        kl_df["lang_freq"] = kl_df["locale"].map(LANG_LOCALE_FREQ).fillna(0)

        df = pd.concat([df, kl_df["lang_freq"]], axis=1)
        return df.drop(columns=["kl"])

    def _extract_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract domain features.
        """
        self.logger.debug("Starting domain feature extraction")

        # extract domain, subdomain, and extension from the url
        domain_df = df["d"].str.lower().apply(get_url_parts).apply(pd.Series).fillna("")

        # simple domain features
        domain_df["domain_length"] = domain_df["domain"].str.len()
        domain_df["domain_entropy"] = domain_df["domain"].apply(entropy_scipy)
        domain_df["domain_digit_ratio"] = (
            domain_df["domain"].str.count(r"\d").div(domain_df["domain"].str.len())
        ).fillna(0)

        # embeddings via fasttext
        self.logger.debug("Computing FastText embeddings for domains")
        try:
            domain_embeddings = (
                domain_df["domain"]
                .fillna("")
                .apply(lambda x: self.fasttext_model.get_word_vector(x))
            )
            domain_matrix = np.vstack(domain_embeddings.values)
            domain_embedding_df = pd.DataFrame(domain_matrix, index=domain_df.index)
            domain_embedding_df.columns = [
                f"domain_ft_dim_{i}" for i in range(domain_embedding_df.shape[1])
            ]
            self.logger.debug(
                f"Created {domain_embedding_df.shape[1]} domain embedding features"
            )
        except Exception as e:
            self.logger.error(f"Error computing domain embeddings: {e}")
            raise

        # simple subdomain features
        domain_df["has_subdomain"] = (~domain_df["subdomain"].isna()).astype(int)
        domain_df["subdomain_entropy"] = domain_df["subdomain"].apply(entropy_scipy)

        # extension features
        domain_df["extension_more_than_one"] = (
            domain_df["extension"].str.count("\\.") > 0
        ).astype(int)
        domain_df["extension_entropy"] = domain_df["extension"].apply(entropy_scipy)

        domain_df["has_country_tld"] = (
            domain_df["extension"]
            .str.split(".")
            .apply(lambda x: any(elem in COUNTRY_CODES for elem in x))
        ).astype(int)

        return pd.concat(
            [
                df.drop("d", axis=1),
                domain_df.drop(["domain", "subdomain", "extension"], axis=1),
                domain_embedding_df,
            ],
            axis=1,
        )

    def _process_query(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process query.
        """
        self.logger.debug("Starting query processing")

        q_df = pd.DataFrame(index=df.index)
        # decode query
        q_df["q_decoded"] = df["q"].apply(lambda x: unquote(x) if pd.notna(x) else x)

        # length and entropy of the query
        q_df["q_length"] = q_df["q_decoded"].str.len()
        q_df["q_entropy"] = q_df["q_decoded"].apply(entropy_scipy)
        q_df["q_word_count"] = q_df["q_decoded"].str.split().apply(len)

        # everything but letters and speces, e.g., digits, punctuation, etc.
        q_df["q_symbol_ratio"] = (
            q_df["q_decoded"].str.count(r"[^a-zA-Z ]").div(q_df["q_decoded"].str.len())
        )

        # embeddings via fasttext
        self.logger.debug("Computing FastText embeddings for queries")
        try:
            ft_embeddings = q_df["q_decoded"].apply(
                lambda x: self.fasttext_model.get_word_vector(x)
            )
            ft_matrix = np.vstack(ft_embeddings.tolist())
            ft_df = pd.DataFrame(ft_matrix, index=q_df.index)
            ft_df.columns = [f"q_ft_dim_{i}" for i in range(ft_df.shape[1])]
            self.logger.debug(f"Created {ft_df.shape[1]} query embedding features")
        except Exception as e:
            self.logger.error(f"Error computing query embeddings: {e}")
            raise

        return pd.concat(
            [df.drop("q", axis=1), q_df.drop("q_decoded", axis=1), ft_df], axis=1
        )
