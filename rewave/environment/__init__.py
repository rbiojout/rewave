import pandas as pd
from gym.envs.registration import register
from .long_portfolio import PortfolioEnv

from datetime import date


start = date(2017, 1, 1)
end = date(2018, 1, 1)
features= ['close', 'high', 'low']
tickers = ['AAPL','A','MSFT']

# register our enviroment with combinations of input arguments
#df_train = pd.read_hdf('./data/poloniex_30m.hf', key='train')
df_train =None

env_specs_args = [
    dict(id='CryptoPortfolioMLP-v0',
         entry_point='rewave.environment.long_portfolio:PortfolioEnv',
         kwargs=dict(
             output_mode='mlp',
             start_date=start,
             end_date=end,
             features_list = features,
             tickers_list=tickers
         )),
    dict(id='CryptoPortfolioEIIE-v0',
         entry_point='rewave.environment.long_portfolio:PortfolioEnv',
         kwargs=dict(
             output_mode='EIIE',
             start_date=start,
             end_date=end,
             features_list = features,
             tickers_list = tickers
)
         ),
    dict(id='CryptoPortfolioAtari-v0',
         entry_point='rewave.environment.long_portfolio:PortfolioEnv',
         kwargs=dict(
             output_mode='atari',
             start_date=start,
             end_date=end,
             features_list = features,
             tickers_list = tickers
         ))
]
env_specs = [spec['id'] for spec in env_specs_args]

# register our env's on import
for env_spec_args in env_specs_args:
    register(**env_spec_args)
