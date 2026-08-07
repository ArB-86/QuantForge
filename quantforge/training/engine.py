import pandas as pd

class WalkForwardEngine:
    def __init__(self, model_class, model_params, splitter):
        self.model_class = model_class
        self.model_params = model_params
        self.splitter = splitter

    def run(self, df: pd.DataFrame, feature_columns: list, target: str):
        # Ensure Ticker and Date are present as columns for robust .loc selection
        if 'Ticker' not in df.columns or 'Date' not in df.columns:
            df = df.reset_index()
            
        splits = self.splitter.split(df)
        oos_preds = []
        
        for train_idx, test_idx in splits:
            train_df = df.loc[train_idx]
            test_df = df.loc[test_idx, ['Date', 'Ticker', target, 'RET_1D']].copy()
            
            X_train = train_df[feature_columns]
            y_train = train_df[target]
            X_test = df.loc[test_idx, feature_columns]
            
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)
            
            test_df['Raw_Prediction'] = model.predict(X_test)
            test_df['Prediction'] = test_df['Raw_Prediction']
            oos_preds.append(test_df)
            
        if not oos_preds:
            return pd.DataFrame()
            
        return pd.concat(oos_preds, ignore_index=True)
