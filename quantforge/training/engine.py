import pandas as pd
from quantforge.dataset.splitter import WalkForwardSplitter

class WalkForwardEngine:
    def __init__(self, model_class, model_params, splitter):
        self.model_class = model_class
        self.model_params = model_params
        self.splitter = splitter

    def run(self, df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
        splits = self.splitter.split(df)
        oos_preds = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = df.loc[train_idx, features].to_numpy(), df.loc[train_idx, target].to_numpy()
            X_test = df.loc[test_idx, features].to_numpy()
            
            model = self.model_class(self.model_params)
            model.fit(X_train, y_train)
            
            test_df = df.loc[test_idx, ['Date', 'Ticker', target, 'RET_1D']].copy()
            if 'VOL_20D' in df.columns:
                test_df['VOL_20D'] = df.loc[test_idx, 'VOL_20D'].values
                
            test_df['Raw_Prediction'] = model.predict(X_test)
            oos_preds.append(test_df)
            
        final_df = pd.concat(oos_preds, ignore_index=True)
        final_df = final_df.sort_values(['Ticker', 'Date'])
        # SMOOTH THE SIGNAL: 3-day rolling average of predictions to kill daily flip-flopping
        final_df['Prediction'] = final_df.groupby('Ticker')['Raw_Prediction'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        return final_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
