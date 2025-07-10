
import logging
from typing import Optional

import fasttext
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from helpers import parse_url_params_simple

# === CONFIG ===
RANDOM_STATE = 42
SVD_N_COMPONENTS = 4
TTC_BINS = [-np.inf] + [1299.0, 3077.0, 7092.0, 15432.5, 48821.0]+ [np.inf]
TTC_LABELS = [0, 1, 2, 3, 4, 5]
COUNTRY_CODES= ['AF', 'AX', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', 'AR',
       'AM', 'AW', 'AU', 'AT', 'AZ', 'BS', 'BH', 'BD', 'BB', 'BY', 'BE',
       'BZ', 'BJ', 'BM', 'BT', 'BO', 'BQ', 'BA', 'BW', 'BV', 'BR', 'IO',
       'BN', 'BG', 'BF', 'BI', 'CV', 'KH', 'CM', 'CA', 'KY', 'CF', 'TD',
       'CL', 'CN', 'CX', 'CC', 'CO', 'KM', 'CG', 'CD', 'CK', 'CR', 'CI',
       'HR', 'CU', 'CW', 'CY', 'CZ', 'DK', 'DJ', 'DM', 'DO', 'EC', 'EG',
       'SV', 'GQ', 'ER', 'EE', 'SZ', 'ET', 'FK', 'FO', 'FJ', 'FI', 'FR',
       'GF', 'PF', 'TF', 'GA', 'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL',
       'GD', 'GP', 'GU', 'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 'HM', 'VA',
       'HN', 'HK', 'HU', 'IS', 'IN', 'ID', 'IR', 'IQ', 'IE', 'IM', 'IL',
       'IT', 'JM', 'JP', 'JE', 'JO', 'KZ', 'KE', 'KI', 'KP', 'KR', 'KW',
       'KG', 'LA', 'LV', 'LB', 'LS', 'LR', 'LY', 'LI', 'LT', 'LU', 'MO',
       'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH', 'MQ', 'MR', 'MU', 'YT',
       'MX', 'FM', 'MD', 'MC', 'MN', 'ME', 'MS', 'MA', 'MZ', 'MM', 'NA',
       'NR', 'NP', 'NL', 'NC', 'NZ', 'NI', 'NE', 'NG', 'NU', 'NF', 'MK',
       'MP', 'NO', 'OM', 'PK', 'PW', 'PS', 'PA', 'PG', 'PY', 'PE', 'PH',
       'PN', 'PL', 'PT', 'PR', 'QA', 'RE', 'RO', 'RU', 'RW', 'BL', 'SH',
       'KN', 'LC', 'MF', 'PM', 'VC', 'WS', 'SM', 'ST', 'SA', 'SN', 'RS',
       'SC', 'SL', 'SG', 'SX', 'SK', 'SI', 'SB', 'SO', 'ZA', 'GS', 'SS',
       'ES', 'LK', 'SD', 'SR', 'SJ', 'SE', 'CH', 'SY', 'TW', 'TJ', 'TZ',
       'TH', 'TL', 'TG', 'TK', 'TO', 'TT', 'TN', 'TR', 'TM', 'TC', 'TV',
       'UG', 'UA', 'AE', 'GB', 'US', 'UM', 'UY', 'UZ', 'VU', 'VE', 'VN',
       'VG', 'VI', 'WF', 'EH', 'YE', 'ZM', 'ZW'
]
COUNTRY_CODE_FREQ = pd.Series({"DE": 0.2164631543, "GB": 0.1791478098, "FR": 0.0929041072, "CA": 0.0660917086, "NL": 0.0634446247, "IT": 0.0623345573, "ES": 0.0481598497, "IN": 0.0296302621, "RU": 0.0286055845, "SE": 0.0233114166, "AU": 0.018358808, "AT": 0.0133208095, "MX": 0.0130646401, "BR": 0.0118691828, "UG": 0.011356844, "IE": 0.0092220989, "FI": 0.0072581334, "NO": 0.007001964, "ID": 0.0059772863, "SC": 0.0057211169, "SG": 0.0052941679, "CH": 0.0046964392, "DK": 0.0046964392, "PH": 0.0038425412, "ZA": 0.0036717616, "NZ": 0.0028178635, "PL": 0.0023909145, "RO": 0.0023055247, "AE": 0.0021347451, "JP": 0.0020493553, "TH": 0.0019639655, "HK": 0.0018785757, "BD": 0.0018785757, "BE": 0.0017931859, "KE": 0.0017931859, "CZ": 0.0017931859, "MY": 0.0017077961, "IR": 0.0014516267, "GR": 0.0013662369, "CO": 0.0013662369, "MA": 0.0013662369, "NG": 0.0012808471, "PK": 0.0012808471, "SA": 0.0012808471, "LU": 0.0009392878, "TR": 0.000853898, "PT": 0.000853898, "AL": 0.000853898, "VN": 0.000853898, "KR": 0.000853898, "IL": 0.000853898, "UA": 0.0007685082, "MT": 0.0007685082, "BG": 0.0007685082, "DO": 0.0007685082, "EG": 0.0006831184, "LK": 0.0006831184, "BO": 0.0005977286, "SI": 0.0005977286, "SK": 0.0005977286, "RS": 0.0005977286, "LV": 0.0005977286, "MM": 0.0005977286, "PR": 0.0005123388, "AF": 0.0005123388, "CL": 0.000426949, "GH": 0.000426949, "TZ": 0.000426949, "MN": 0.000426949, "AR": 0.000426949, "CN": 0.000426949, "HR": 0.000426949, "LB": 0.000426949, "QA": 0.000426949, "CY": 0.0003415592, "JO": 0.0003415592, "LA": 0.0003415592, "BA": 0.0003415592, "HU": 0.0003415592, "CM": 0.0003415592, "NP": 0.0003415592, "AG": 0.0003415592, "RE": 0.0002561694, "BS": 0.0002561694, "TT": 0.0002561694, "EE": 0.0002561694, "CW": 0.0002561694, "KH": 0.0002561694, "unknown": 0.0002561694, "TN": 0.0002561694, "KW": 0.0002561694, "GE": 0.0002561694, "TW": 0.0002561694, "OM": 0.0002561694, "GG": 0.0002561694, "SV": 0.0001707796, "PA": 0.0001707796, "HN": 0.0001707796, "ET": 0.0001707796, "GT": 0.0001707796, "MU": 0.0001707796, "VE": 0.0001707796, "KG": 0.0001707796, "EC": 0.0001707796, "BH": 0.0001707796, "MZ": 0.0001707796, "CU": 0.0001707796, "DZ": 0.0001707796, "KZ": 0.0001707796, "MK": 0.0001707796, "BB": 0.0001707796, "JM": 0.0001707796, "GY": 0.0001707796, "AM": 8.53898e-05, "MC": 8.53898e-05, "RW": 8.53898e-05, "MV": 8.53898e-05, "MG": 8.53898e-05, "PE": 8.53898e-05, "VI": 8.53898e-05, "MQ": 8.53898e-05, "BM": 8.53898e-05, "MD": 8.53898e-05, "CR": 8.53898e-05, "LI": 8.53898e-05, "VU": 8.53898e-05, "LR": 8.53898e-05, "GQ": 8.53898e-05, "BF": 8.53898e-05, "BW": 8.53898e-05, "PG": 8.53898e-05, "JE": 8.53898e-05, "IQ": 8.53898e-05, "NI": 8.53898e-05, "MF": 8.53898e-05, "PF": 8.53898e-05, "LT": 8.53898e-05, "SN": 8.53898e-05, "CV": 8.53898e-05, "BT": 8.53898e-05, "ZW": 8.53898e-05, "ZM": 8.53898e-05, "HT": 8.53898e-05})

class FeatureTransformation:
    """
    Feature engineering pipeline that replicates all transformations
    from the feature engineering notebook.
    """

    # column mappings
    CATEGORICAL_COLUMNS = ['browser', 'device', 'region']
    COLUMNS_TO_DROP = ['st','n', 'f', 'adx_name', 'atbexp', 'va', 'atbva']
    SPARSE_COLUMNS_TO_ENCODE = ['r', 'kp', 'sld']
    SPARSE_COLUMNS_TO_COMBINE = ['lsexp1', 'om', 'adx', 'bkl', 'atb']
    
    # replace fake nulls to np.nan
    FAKE_NULLS_TO_NAN = ['null', 'none']

    # column type transformations
    COLUMNS_TO_INT = ['nt']

    # input columns
    # input columns; expected all as str
    INPUT_COLUMNS = ['datetime', 'browser', 'device', 'region', 'url']

    def __init__(self, fasttext_model_path: str = "models/cc.en.300.bin", log_level: str = "INFO"):
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
            logging.error(f"Error loading FastText model: {e}")
            raise e

        
        self.logger.info("FeatureTransformation initialized")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input dataframe.
        """
        # Step 0: Unnest data
        df_transformed = self._clean_data(df)

        # Step 1: One-hot encode categorical features
        df_transformed = self._encode_categorical_features(df_transformed)

        # Step 2: Drop constant columns
        df_transformed = self._drop_constant_columns(df_transformed)

        # Step 3: One-hot encode categorical features with missing values
        df_transformed = self._encode_categorical_features(df_transformed, drop_first=False, missing_values=True, values_to_missing=self.FAKE_NULLS_TO_NAN)

        # Step 4: Combine sparse features
        df_transformed = self._combine_sparse_features(df_transformed)
        
        # Step 5: Process time-to-click: log and bin
        df_transformed = self._process_ttc(df_transformed)

        # Step 6: Process country code

        return df_transformed



    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Unnest ad click url params and extract hour of day.
        """
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.loc[:, 'hour'] = df['datetime'].dt.hour
        df.loc[:, 'params_dict'] = df['url'].apply(parse_url_params_simple)
        df = df.join(df['params_dict'].apply(pd.Series))

        # direct trasnformations
        df[self.COLUMNS_TO_INT] = df[self.COLUMNS_TO_INT].astype(int)

        return df.drop(columns=['url', 'datetime', 'params_dict'])

    def _encode_categorical_features(
        self,
        df: pd.DataFrame,
        drop_first: bool = True,
        missing_values: bool = False,
        values_to_missing: Optional[list] = None
    ) -> pd.DataFrame:
        """
        One-hot encode categorical features.
        If drop_first is True, drop first category to avoid multicollinearity in linear models, thus returns len(CATEGORICAL_COLUMNS) less columns.
        If missing_values is True, encode missing values as a separate category.
        If values_to_missing is provided, replace these values with np.nan.
        """
        kwargs = {}
        if drop_first:
            kwargs['drop_first'] = True
        if missing_values:
            kwargs['dummy_na'] = True
        if values_to_missing:
            df[self.CATEGORICAL_COLUMNS] = df[self.CATEGORICAL_COLUMNS].replace(values_to_missing, np.nan)

        encoded_df = pd.get_dummies(df, columns=self.CATEGORICAL_COLUMNS, **kwargs)
        return pd.concat([df.drop(self.CATEGORICAL_COLUMNS, axis=1), encoded_df], axis=1)

    def _drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that are constant.
        """
        return df.drop(columns=self.COLUMNS_TO_DROP)

    def _combine_sparse_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine sparse features with truncated singular value decomposition.
        """

        # get sparse features - ensure it's a DataFrame
        sparse_df = df[self.SPARSE_COLUMNS_TO_ENCODE]
        if isinstance(sparse_df, pd.Series):
            sparse_df = sparse_df.to_frame()
        
        # encode sparse features
        sparse_dummies = self._encode_categorical_features(
            sparse_df, 
            drop_first=False, 
            missing_values=True, 
            values_to_missing=self.FAKE_NULLS_TO_NAN
        )

        svd = TruncatedSVD(
            n_components=SVD_N_COMPONENTS, # must be less than or equal to the number of features
            random_state=RANDOM_STATE
        )

        svd_features = svd.fit_transform(sparse_dummies)

        svd_df = pd.DataFrame(
            svd_features, 
            columns=[f'sparse_{i}' for i in range(SVD_N_COMPONENTS)],  # type: ignore
            index=df.index
        )

        return pd.concat([df.drop(columns=self.SPARSE_COLUMNS_TO_COMBINE, axis=1), svd_df], axis=1)


    def _process_ttc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Log transform time-to-click and bin it.
        """
        df['ttc_log'] = np.log1p(df['ttc'].astype(float)) 
        df['ttc_bucket'] = pd.cut(df['ttc'].astype(float), bins=TTC_BINS, labels=TTC_LABELS)
        return df.drop(columns=['ttc'])

    def _process_country_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process country code.
        """

        is_country = df['ct'].str.upper().fillna('').isin(COUNTRY_CODES)
        if sum(~is_country) > 0:
            df.loc[~is_country, 'ct'] = 'unknown'

        df['ct_freq'] = df['ct'].map(COUNTRY_CODE_FREQ)

        return df.drop(columns=['ct'])



if __name__ == "__main__":
    transform = FeatureTransformation()
    print(transform.fasttext_model)