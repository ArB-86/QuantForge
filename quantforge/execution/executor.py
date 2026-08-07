import pandas as pd

class OrderExecutor:
    def __init__(self, portfolio_value=1000000.0):
        self.portfolio_value = portfolio_value
        # Realistic Indian market transaction friction model (approximate basis points)
        self.brokerage_fee_pct = 0.0003  # 0.03%
        self.stt_tax_pct = 0.001        # 0.1% on delivery sell/buy side average
        self.exchange_charges_pct = 0.0000325 # NSE turnover charge
        self.gst_pct = 0.18             # 18% GST on brokerage + exchange charges

    def generate_orders(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        if portfolio_df.empty:
            print('Portfolio dataframe is empty. No orders to generate.')
            return pd.DataFrame()

        print(f'Generating institutional orders for Portfolio Value: ?{self.portfolio_value:,.2f}')
        
        orders = []
        for _, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            weight = row['Weight']
            
            capital_allocated = self.portfolio_value * weight
            price = row.get('Close', row.get('Execution_Price', 0.0))
            if pd.isna(price) or price <= 0:
                raise ValueError(f'Fatal: Invalid execution price for {ticker}. Halting to prevent bad fills.')
                
            target_shares = int(capital_allocated // price)
            if target_shares <= 0:
                target_shares = 1
                
            notional_value = target_shares * price
            
            # Compute realistic friction costs
            brokerage = notional_value * self.brokerage_fee_pct
            exchange = notional_value * self.exchange_charges_pct
            gst = (brokerage + exchange) * self.gst_pct
            stt = notional_value * self.stt_tax_pct
            total_friction = brokerage + exchange + gst + stt
            
            orders.append({
                'Ticker': ticker,
                'Target_Shares': target_shares,
                'Execution_Price': price,
                'Notional_Capital': notional_value,
                'Estimated_Friction_Cost': total_friction,
                'Target_Weight': weight
            })
            
        orders_df = pd.DataFrame(orders)
        print(f'Total Estimated Friction/Taxes for Batch: ?{orders_df["Estimated_Friction_Cost"].sum():,.2f}')
        return orders_df
