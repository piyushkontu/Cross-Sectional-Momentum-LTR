# Cross-Sectional-Momentum-LTR
This repository contains the code to replicate the paper Poh, Daniel and Lim, Bryan and Zohren, Stefan and Roberts, Stephen, Building Cross-Sectional Systematic Strategies By Learning to Rank (December 12, 2020) , Available at SSRN: https://ssrn.com/abstract=3751012

The paper uses 22 momentum indicators described below -

1. Raw returns -
	3m, 6m, 12m raw returns (3 indicators)

2. Normalized returns -
	3m, 6m, 12m raw returns normalized by volatility and scaled by time period (3 indicators)

3. MACD indicators
	Short windows - [8,16,32]
	Long windows - [24,48,96]
	
	1. One Final MACD indicator = sum of MACD indicators for the 3 windows above (1 indicator)
	2. 3 individual MACD indicators for the 3 windows above (3 indicators)
	3. Individual MACD indicators for each window lagged by 1m,3m,6m and 1y. (3 x 4 indicators)
