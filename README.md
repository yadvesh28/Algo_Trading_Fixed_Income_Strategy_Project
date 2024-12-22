# Sovereign Bonds Analysis Strategy Project (FIN 554)

A sophisticated quantitative analysis framework for sovereign bond markets that combines machine learning techniques with traditional financial indicators. This research explores the predictive power of advanced modeling techniques while incorporating both technical and macroeconomic factors to develop robust trading strategies.

---
## Team

<div align="center">
<img src="imgages/ganesh_img.jpg" width="200" height="auto" alt="Ganesh Ashwin Vadari Venkata">
<p><strong>Ganesh Ashwin Vadari Venkata</strong></p>
</div>

Email: [gav3@illinois.edu](mailto:gav3@illinois.edu)  
LinkedIn: [https://www.linkedin.com/in/ganeshashwinvv/](https://www.linkedin.com/in/ganeshashwinvv/)

I am a Master's candidate in Financial Mathematics at the University of Illinois Urbana-Champaign, specializing in quantitative finance and advanced analytics for financial decision-making. With a Chartered Accountant certification and professional expertise in corporate development and financial advisory, I have optimized processes and delivered impactful results in valuation and strategic finance working with the CFO office at Cavinkare and complex accounting advisory branch at Grant Thornton.
---

<div align="center">
<img src="images/gaurav_img.jpeg" width="200" height="auto" alt="Gaurav Ghosh">
<p><strong>Gaurav Ghosh</strong></p>
</div>

Email: [gauravg4@illinois.edu](mailto:gauravg4@illinois.edu)  
LinkedIn: [https://www.linkedin.com/in/gauravg29/](https://www.linkedin.com/in/gauravg29/)

I am a Master's candidate in Finance at the University of Illinois Urbana-Champaign, with expertise in fixed-income analysis, portfolio risk monitoring, and financial modeling. A Chartered Financial Analyst Level II candidate, i possess advanced technical skills in Python, R, and Tableau with experience in regulatory compliance, auditing, and options trading. I am passionate about leveraging data-driven insights to enhance investment strategies and optimize portfolio performance under diverse market conditions.

---

<div align="center">
<img src="images/krish_img.jpeg" width="200" height="auto" alt="Krish Desai">
<p><strong>Krish Desai</strong></p>
</div>

Email: [kcdesai2@illinois.edu](mailto:kcdesai2@illinois.edu)  
LinkedIn: [https://www.linkedin.com/in/krish-desai-4447971b3](https://www.linkedin.com/in/krish-desai-4447971b3)

A highly motivated  individual with a strong academic background in financial mathematics and computer science. Eager to contribute technical expertise and contribute to innovative solutions. A quick learner with a strong work ethic and a collaborative spirit, always seeking a challenging problem in a dynamic environment.

---

<div align="center">
<img src="img/DSC05522.JPG" width="200" height="auto" alt="Yadvesh Yadav">
<p><strong>Yadvesh Yadav</strong></p>
</div>

Email: [yyada@illinois.edu](mailto:yyada@illinois.edu)  
LinkedIn: [https://www.linkedin.com/in/yadvesh/](https://www.linkedin.com/in/yadvesh/)

As a Master's student in Financial Mathematics at the University of Illinois Urbana-Champaign, with a background as a Data Science Engineer, I specialize in financial data analysis, predictive modeling, and algorithmic trading. My passion lies in leveraging mathematical and computational techniques to develop innovative, quantitative trading strategies and solutions.

---

## Project Overview

Our research investigates whether machine learning models, specifically Lasso and LightGBM, can outperform traditional forecasting methods for sovereign 10-Year bonds across different countries. The approach integrates a diverse set of inputs, from macroeconomic variables to momentum-based technical indicators, creating a comprehensive analytical framework for bond yield prediction.

### Research Foundations

The project builds upon existing literature in both bond-based and stock-based studies. Notable influences include Dubrov's (2015) work on Monte Carlo simulations for exotic bond pricing and Ganguli and Dunnmon's (2017) extensive study of U.S. corporate bond price prediction. We extend these approaches by incorporating modern machine learning techniques and developing a more sophisticated signal generation framework.

Our methodology addresses several critical hypotheses about market behavior and model performance. The primary investigation focuses on whether preprocessing techniques such as winsorizing and exponential smoothing can significantly improve model performance when applied to sovereign bond datasets. This is particularly relevant given the complex nature of bond market dynamics and the presence of multiple market regimes.

## Technical Implementation

### Data Integration Framework

The analysis incorporates data from multiple authoritative sources to create a comprehensive view of market conditions. For macroeconomic indicators, we utilize the Federal Reserve Economic Data (FRED) database, collecting critical metrics including:

Consumer Price Index (CPI) data provides insight into inflation trends, while Producer Price Index (PPI) offers early signals of price pressures in the production pipeline. The Federal Funds Rate and Unemployment Rate data help contextualize monetary policy and economic conditions. We pay particular attention to the 10-Year Minus 2-Year Treasury Spread as a leading indicator of economic cycles.

For sovereign bond yields, we source data from CRSP covering October 1993 through June 2018, providing a robust historical dataset for model training and validation. The Australian market analysis uses data from Investing.com spanning January 1992 to December 2023, enabling cross-market comparison and strategy validation.

### Model Architecture

The heart of our analysis lies in the sophisticated interplay between multiple modeling approaches. Our Lasso regression implementation operates within an expanding window framework, allowing for dynamic feature selection that adapts to changing market conditions. This is complemented by LightGBM's capability to capture non-linear relationships in the data.

The feature selection process incorporates multiple validation techniques:

```python
def validate_features(df, features, window_size):
    """
    Comprehensive feature validation using multiple statistical tests
    """
    results = {}
    for feature in features:
        # Granger causality testing
        gc_result = granger_causality_test(df[['yield', feature]], maxlag=window_size)
        
        # Cointegration analysis for macro indicators
        if feature in macro_indicators:
            coint_result = coint_johansen(df[['yield', feature]], det_order=0, k_ar_diff=window_size)
            
        results[feature] = {
            'granger_p_value': gc_result[0]['ssr_ftest'][1],
            'cointegration_stat': coint_result.lr1[0] if feature in macro_indicators else None
        }
    
    return results
```

### Regime Detection and Signal Generation

A key innovation in our approach is the implementation of a Hidden Markov Model for regime detection. The model uses a two-state framework to classify market conditions into high and low volatility regimes, enabling more nuanced strategy adaptation:

```python
class RegimeDetector:
    def __init__(self, n_regimes=2):
        self.model = GaussianHMM(n_components=n_regimes, covariance_type="diag")
        
    def fit_transform(self, returns, volatility):
        X = np.column_stack([returns, volatility])
        self.model.fit(X)
        states = self.model.predict(X)
        
        # Classify regimes based on volatility levels
        volatility_by_state = [volatility[states == i].mean() for i in range(self.n_regimes)]
        self.low_vol_state = np.argmin(volatility_by_state)
        
        return states
```

For signal generation, we implement a sophisticated Ichimoku Cloud analysis system that adapts to detected market regimes. The system calculates key components including Tenkan-sen (Conversion Line), Kijun-sen (Base Line), and Senkou Spans A and B, with parameters optimized for bond market characteristics.

## Performance Analysis

Our model evaluation reveals several interesting patterns in prediction accuracy across different market regimes. In the low volatility regime, the Lasso model achieved an MSE of 0.0010, significantly outperforming the high volatility regime's MSE of 0.0016. The overall model demonstrated robust performance with an MSE of 0.0015 across all market conditions.

Signal generation statistics show promising directional accuracy, with 31 total signals generated during the testing period. While the raw directional accuracy of 0.4839 might seem modest, the strategy's precision of 0.5000 and recall of 0.7500 suggest effective signal filtering. The F1 Score of 0.6000 indicates a balanced trade-off between precision and recall.

### Australian Market Extension

The model's application to the Australian sovereign bond market provides valuable insights into cross-market applicability. With a directional accuracy of 0.5484 and improved precision metrics (0.5600), the strategy shows potential for geographical expansion. The higher recall rate of 0.8235 in the Australian market suggests particularly effective signal generation in this context.

## Future Development

Our research points to several promising avenues for future enhancement. The current implementation would benefit from more sophisticated parameter optimization techniques, particularly for the Ichimoku Cloud components. We also see potential in developing more advanced regime detection methods that could incorporate additional market factors beyond volatility.

The integration of alternative data sources and more sophisticated machine learning architectures could further improve predictive accuracy. Additionally, expanding the cross-market analysis to other sovereign bond markets could provide valuable insights into the strategy's generalizability.

## Technical Requirements

The implementation requires Python 3.7 or higher and relies on several specialized libraries for financial analysis and machine learning. Key dependencies include pandas for data manipulation, scikit-learn for core machine learning functionality, lightgbm for gradient boosting, and hmmlearn for regime detection.

## Acknowledgments

This research was conducted as part of the FIN 554 course at the University of Illinois Urbana-Champaign. The authors gratefully acknowledge the support and guidance received throughout the project.