## Based on sovereign_final_data from Rep project, this is for 3, 10 , 30 Yr
## basic moving average, rsi, bollinger bands, MACD component code and plotting code

def add_moving_average(df, window):
    for term in ['3 Yr', '10 Yr', '30 Yr']:
        df[f'{term}_MA_{window}'] = df[term].rolling(window=window).mean()
    return df

def add_rsi(df, window):
    for term in ['3 Yr', '10 Yr', '30 Yr']:
        delta = df[term].diff(1)
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        df[f'{term}_RSI_{window}'] = 100 - (100 / (1 + rs))
        scaler_rsi = StandardScaler()
        df[f'{term}_RSI_Scaled_{window}'] = scaler_rsi.fit_transform(df[f'{term}_RSI_{window}'].values.reshape(-1, 1))
        df.drop(f'{term}_RSI_{window}', axis = 1, inplace = True)
    return df

def add_bollinger_bands(df, window):
    for term in ['3 Yr', '10 Yr', '30 Yr']:
        rolling_mean = df[term].rolling(window).mean()
        rolling_std = df[term].rolling(window).std()
        df[f'{term}_Bollinger_High_{window}'] = rolling_mean + (rolling_std * 2)
        df[f'{term}_Bollinger_Low_{window}'] = rolling_mean - (rolling_std * 2)
    return df


# %%
## As we saw above there are no nulls in data from yahoo finance so we'll use that for the 30 Yr treasurey bond.
sovereign_data_final = merged_sovereign_data.drop(['30 Yr'], axis=1)

sovereign_data_final.head()

# %%
sovereign_data_final.rename(columns={"30 Yr YF": "30 Yr"}, inplace=True)

def add_macd(df, slow=26, fast=12, signal=9):
    for term in [ '10 Yr']:
        # Calculate the Fast and Slow EMAs
        ema_fast = df[term].ewm(span=fast, adjust=False).mean()
        ema_slow = df[term].ewm(span=slow, adjust=False).mean()
        
        # Calculate the MACD Line
        macd_line = ema_fast - ema_slow
        df[f'{term}_MACD_Line'] = macd_line
        
        # Calculate the Signal Line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        df[f'{term}_MACD_Signal'] = signal_line
        
        # Calculate the MACD Histogram
        macd_histogram = macd_line - signal_line
        df[f'{term}_MACD_Hist'] = macd_histogram
    return df

sovereign_data_final = add_macd(sovereign_data_final)
sovereign_data_final.dropna(inplace=True) 
sovereign_data_final.head()

# %%
def plot_macd(df, term):
    plt.figure(figsize=(14, 7))
    
    ax1 = plt.subplot(211)
    ax1.plot(df.index, df[term], label=f'{term} Yield', color='blue')
    ax1.set_title(f'{term} Treasury Yield')
    ax1.set_ylabel('Yield (%)')
    ax1.legend(loc='upper left')
    
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(df.index, df[f'{term}_MACD_Line'], label='MACD Line', color='red')
    ax2.plot(df.index, df[f'{term}_MACD_Signal'], label='Signal Line', color='green')
    ax2.bar(df.index, df[f'{term}_MACD_Hist'], label='Histogram', color='grey', alpha=0.3)
    ax2.set_title(f'{term} MACD')
    ax2.set_ylabel('MACD Value')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

plot_macd(sovereign_data_final,'10 Yr')

# %%
# Define the file path
inflation_file_path = r'D:\_IncrementalMaybe\AlgoTrading\Other_Sovereign_Data\CPI_monthly.csv'

# Read the CSV file
inflation_df = pd.read_csv(inflation_file_path)

# Convert 'DATE' column to datetime
inflation_df['DATE'] = pd.to_datetime(inflation_df['DATE'])

# Set 'DATE' as the index
inflation_df.set_index('DATE', inplace=True)

# Rename the inflation column for clarity (optional)
inflation_df.rename(columns={'MEDCPIM158SFRBCLE': 'Inflation_Rate'}, inplace=True)

print(inflation_df.head())

start_date = '1993-10-01'
end_date = '2018-12-31'
inflation_df = inflation_df.loc[start_date:end_date]

daily_index = pd.date_range(start=start_date, end=end_date, freq='D')

# Reindex the DataFrame to include all daily dates
inflation_daily_ffill = inflation_df.reindex(daily_index)

# Forward-fill the missing values
inflation_daily_ffill.fillna(method='ffill', inplace=True)

# Rename the index to 'Date'
inflation_daily_ffill.index.name = 'Date'

# Display the first few rows
print(inflation_daily_ffill.head())

# %%
ffr_file_path = r'D:\_IncrementalMaybe\AlgoTrading\Macro_Data\FEDFUNDS.csv'

ffr_df = pd.read_csv(ffr_file_path)
ffr_df['DATE'] = pd.to_datetime(ffr_df['DATE'])

# Set 'DATE' as the index
ffr_df.set_index('DATE', inplace=True)

ffr_df.head()

unemp_rate_file_path = r'D:\_IncrementalMaybe\AlgoTrading\Macro_Data\UNRATE.csv'

unemp_rate_df = pd.read_csv(unemp_rate_file_path)
unemp_rate_df['DATE'] = pd.to_datetime(unemp_rate_df['DATE'])

# Set 'DATE' as the index
unemp_rate_df.set_index('DATE', inplace=True)

unemp_rate_df.head()


# %%

ma_window = 30
rsi_window = 30
bollinger_window = 30

# Add Moving Average
sovereign_data_final = add_moving_average(sovereign_data_final, ma_window)

# Add RSI
sovereign_data_final = add_rsi(sovereign_data_final, rsi_window)

# Add Bollinger Bands
sovereign_data_final = add_bollinger_bands(sovereign_data_final, bollinger_window)
sovereign_data_final.dropna(inplace=True) 
sovereign_data_final.head()

# %%
sovereign_data_final.info()

# %%
## Now craeting lagged features to do a time series analysis for different models
plt.figure(figsize=(12, 6))
plt.plot(sovereign_data_final.index, sovereign_data_final['10 Yr'], label='10 Yr Yield')
plt.plot(sovereign_data_final.index, sovereign_data_final['10 Yr_MA_30'], label='10 Yr MA (30)')
plt.title('10 Year Treasury Yield and 30-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Yield (%)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(sovereign_data_final.index, sovereign_data_final['10 Yr'], label='10 Yr Yield')
plt.plot(sovereign_data_final.index, sovereign_data_final['10 Yr_Bollinger_High_30'], label='10 Yr Bollinger_High (30)')
plt.plot(sovereign_data_final.index, sovereign_data_final['10 Yr_Bollinger_Low_30'], label='10 Yr Bollinger_Low (30)')
plt.title('10 Year Treasury Yield and 30-Day RSI')
plt.xlabel('Date')
plt.ylabel('Yield (%)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(sovereign_data_final.index, sovereign_data_final['10 Yr'], label='10 Yr Yield')
plt.plot(sovereign_data_final.index, sovereign_data_final['10 Yr_RSI_Scaled_30'], label='10 Yr RSI (30)')
plt.title('10 Year Treasury Yield and 30-Day RSI')
plt.xlabel('Date')
plt.ylabel('Yield (%)')
plt.legend()
plt.show()