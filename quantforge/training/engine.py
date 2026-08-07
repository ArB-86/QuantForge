import pandas as pd

class WalkForwardEngine:
    def __init__(self, model_class, model_params, splitter):
        self.model_class = model_class
        self.model_params = model_params
        self.splitter = splitter

    def run(self, df: pd.DataFrame, feature_columns: list, target: str):
        if isinstance(df.index, pd.MultiIndex) or df.index.name is not None or any(col in ['Ticker', 'Date'] for col in df.index.names if col):
            df = df.reset_index()
            
        for col in ['index', 'level_0', 'level_1']:
            if col in df.columns:
                df = df.drop(columns=[col])
                
        df.columns = [str(c).strip() for c in df.columns]
        
        if 'Ticker' not in df.columns or 'Date' not in df.columns:
            raise KeyError(f"Fatal: 'Ticker' or 'Date' missing from DataFrame columns. Available columns: {list(df.columns)}")

        splits = self.splitter.split(df)
        oos_preds = []
        
        for train_idx, test_idx in splits:
            train_df = df.iloc[train_idx]
            cols_to_keep = ['Date', 'Ticker', target, 'RET_1D']
            if 'VOL_20D' in df.columns and 'VOL_20D' not in cols_to_keep:
                cols_to_keep.append('VOL_20D')
                
            test_df = df.iloc[test_idx][cols_to_keep].copy()
            
            X_train = train_df[feature_columns]
            y_train = train_df[target]
            X_test = df.iloc[test_idx][feature_columns]
            
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)
            
            test_df['Raw_Prediction'] = model.predict(X_test)
            test_df['Prediction'] = test_df['Raw_Prediction']
            oos_preds.append(test_df)
            
        if not oos_preds:
            return pd.DataFrame()
            
        return pd.concat(oos_preds, ignore_index=True)
