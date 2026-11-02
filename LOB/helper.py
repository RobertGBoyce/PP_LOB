import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def this_sqrt(x):
    return math.sqrt(max(x,0.0))

# Apply the function using np.vectorize
this_vec_sqrt = np.vectorize(this_sqrt)

def expw_zscore(df : pd.DataFrame, lookback : int = 501) -> pd.DataFrame:
    eps = 10**(-6)

    lookback_decay_value = 0.01 # lambda**lookback = lookback_decay_value
    this_lambda = math.exp( math.log(lookback_decay_value) / lookback)
    this_alpha = 1.0 - this_lambda

    #df_intermediate = pd.DataFrame() # empty dataframe
    df_output = pd.DataFrame() # empty dataframe

    these_columns = list(df.columns).copy()
    print(these_columns)
    for col_name in these_columns:
        print(col_name)
        mu_col = col_name + "_mu"
        np_mu_col = df[col_name].ewm(alpha=this_alpha, adjust=False).mean().shift(1)
        np_mu_col[0] = df[col_name].iloc[0]

        # V_0 = 0,  V_{i+1} = lam*V_i + (1-lam)*(x_i - mu_i)**2
        sq_dev = (df[col_name] - np_mu_col) ** 2
        #V_col = col_name + "_V"
        np_V_col = sq_dev.ewm(alpha=alpha, adjust=False).mean().shift(1)
        np_V_col[0] = 0.0

        #eps = 10**(-6)
        #sigma_col = col_name + "_sigma"
        np_sigma_col = this_vec_sqrt(np_V_col + eps)

        z_col = col_name + "_z"
        np_z_col = (df[col_name] - np_mu_col) / np_sigma_col

        df_output[z_col] = np_z_col[lookback:]

    return df_output
    
def read_and_format_orderbook(orderbook_file : str = "UNDEFINED") -> pd.DataFrame:
    
    df_ob=pd.read_csv(orderbook_file)

    # Add column names
    df_ob.columns=[
    "AskPrice1",
    "AskSize1",
    "BidPrice1",
    "BidSize1",
    "AskPrice2",
    "AskSize2",
    "BidPrice2",
    "BidSize2",
    "AskPrice3",
    "AskSize3",
    "BidPrice3",
    "BidSize3",
    "AskPrice4",
    "AskSize4",
    "BidPrice4",
    "BidSize4",
    "AskPrice5",
    "AskSize5",
    "BidPrice5",
    "BidSize5",
    "AskPrice6",
    "AskSize6",
    "BidPrice6",
    "BidSize6",
    "AskPrice7",
    "AskSize7",
    "BidPrice7",
    "BidSize7",
    "AskPrice8",
    "AskSize8",
    "BidPrice8",
    "BidSize8",
    "AskPrice9",
    "AskSize9",
    "BidPrice9",
    "BidSize9",
    "AskPrice10",
    "AskSize10",
    "BidPrice10",
    "BidSize10",
    ]
    
    return df_ob
    
def generate_orderbook_price_diffs(df_ob_price_levels : pd.DataFrame, 
                                   min_index : int = 0,
                                   max_index : int = -1
                                  ) -> pd.DataFrame:

    ob_AP1 = df_ob_price_levels['AskPrice1'] [min_index : max_index]
    ob_BP1 = df_ob_price_levels['BidPrice1'][min_index : max_index]

    ob_AP2 = df_ob_price_levels['AskPrice2'][min_index : max_index]
    ob_BP2 = df_ob_price_levels['BidPrice2'][min_index : max_index]

    ob_AP3 = df_ob_price_levels['AskPrice3'][min_index : max_index]
    ob_BP3 = df_ob_price_levels['BidPrice3'][min_index : max_index]

    ob_AP4 = df_ob_price_levels['AskPrice4'][min_index : max_index]
    ob_BP4 = df_ob_price_levels['BidPrice4'][min_index : max_index]

    ob_AP5 = df_ob_price_levels['AskPrice5'][min_index : max_index]
    ob_BP5 = df_ob_price_levels['BidPrice5'][min_index : max_index]

    ob_AP6 = df_ob_price_levels['AskPrice6'][min_index : max_index]
    ob_BP6 = df_ob_price_levels['BidPrice6'][min_index : max_index]

    ob_AP7 = df_ob_price_levels['AskPrice7'][min_index : max_index]
    ob_BP7 = df_ob_price_levels['BidPrice7'][min_index : max_index]

    ob_AP8 = df_ob_price_levels['AskPrice8'][min_index : max_index]
    ob_BP8 = df_ob_price_levels['BidPrice8'][min_index : max_index]

    ob_AP9 = df_ob_price_levels['AskPrice9'][min_index : max_index]
    ob_BP9 = df_ob_price_levels['BidPrice9'][min_index : max_index]

    ob_AP10 = df_ob_price_levels['AskPrice10'][min_index : max_index]
    ob_BP10 = df_ob_price_levels['BidPrice10'][min_index : max_index]

    df_ob_price_diffs = pd.DataFrame() # empty dataframe

    ob_AP2_minus_ob_AP1 = ob_AP2 - ob_AP1
    ob_AP3_minus_ob_AP1 = ob_AP3 - ob_AP1
    ob_AP4_minus_ob_AP1 = ob_AP4 - ob_AP1
    ob_AP5_minus_ob_AP1 = ob_AP5 - ob_AP1
    ob_AP6_minus_ob_AP1 = ob_AP6 - ob_AP1
    ob_AP7_minus_ob_AP1 = ob_AP7 - ob_AP1
    ob_AP8_minus_ob_AP1 = ob_AP8 - ob_AP1
    ob_AP9_minus_ob_AP1 = ob_AP9 - ob_AP1
    ob_AP10_minus_ob_AP1 = ob_AP10 - ob_AP1

    df_ob_price_diffs["AP2_minus_AP1"] = ob_AP2_minus_ob_AP1
    df_ob_price_diffs["AP3_minus_AP1"] = ob_AP3_minus_ob_AP1
    df_ob_price_diffs["AP4_minus_AP1"] = ob_AP4_minus_ob_AP1
    df_ob_price_diffs["AP5_minus_AP1"] = ob_AP5_minus_ob_AP1
    df_ob_price_diffs["AP6_minus_AP1"] = ob_AP6_minus_ob_AP1
    df_ob_price_diffs["AP7_minus_AP1"] = ob_AP7_minus_ob_AP1
    df_ob_price_diffs["AP8_minus_AP1"] = ob_AP8_minus_ob_AP1
    df_ob_price_diffs["AP9_minus_AP1"] = ob_AP9_minus_ob_AP1
    df_ob_price_diffs["AP10_minus_AP1"] = ob_AP10_minus_ob_AP1

    ob_AP1_minus_ob_BP1 = ob_AP1 - ob_BP1
    ob_AP2_minus_ob_BP1 = ob_AP2 - ob_BP1
    ob_AP3_minus_ob_BP1 = ob_AP3 - ob_BP1
    ob_AP4_minus_ob_BP1 = ob_AP4 - ob_BP1
    ob_AP5_minus_ob_BP1 = ob_AP5 - ob_BP1
    ob_AP6_minus_ob_BP1 = ob_AP6 - ob_BP1
    ob_AP7_minus_ob_BP1 = ob_AP7 - ob_BP1
    ob_AP8_minus_ob_BP1 = ob_AP8 - ob_BP1
    ob_AP9_minus_ob_BP1 = ob_AP9 - ob_BP1
    ob_AP10_minus_ob_BP1 = ob_AP10 - ob_BP1

    df_ob_price_diffs["AP1_minus_BP1"] = ob_AP1_minus_ob_BP1
    df_ob_price_diffs["AP2_minus_BP1"] = ob_AP2_minus_ob_BP1
    df_ob_price_diffs["AP3_minus_BP1"] = ob_AP3_minus_ob_BP1
    df_ob_price_diffs["AP4_minus_BP1"] = ob_AP4_minus_ob_BP1
    df_ob_price_diffs["AP5_minus_BP1"] = ob_AP5_minus_ob_BP1
    df_ob_price_diffs["AP6_minus_BP1"] = ob_AP6_minus_ob_BP1
    df_ob_price_diffs["AP7_minus_BP1"] = ob_AP7_minus_ob_BP1
    df_ob_price_diffs["AP8_minus_BP1"] = ob_AP8_minus_ob_BP1
    df_ob_price_diffs["AP9_minus_BP1"] = ob_AP9_minus_ob_BP1
    df_ob_price_diffs["AP10_minus_BP1"] = ob_AP10_minus_ob_BP1

    ob_BP2_minus_ob_BP1 = ob_BP2 - ob_BP1
    ob_BP3_minus_ob_BP1 = ob_BP3 - ob_BP1
    ob_BP4_minus_ob_BP1 = ob_BP4 - ob_BP1
    ob_BP5_minus_ob_BP1 = ob_BP5 - ob_BP1
    ob_BP6_minus_ob_BP1 = ob_BP6 - ob_BP1
    ob_BP7_minus_ob_BP1 = ob_BP7 - ob_BP1
    ob_BP8_minus_ob_BP1 = ob_BP8 - ob_BP1
    ob_BP9_minus_ob_BP1 = ob_BP9 - ob_BP1
    ob_BP10_minus_ob_BP1 = ob_BP10 - ob_BP1

    df_ob_price_diffs["BP2_minus_BP1"] = ob_BP2_minus_ob_BP1
    df_ob_price_diffs["BP3_minus_BP1"] = ob_BP3_minus_ob_BP1
    df_ob_price_diffs["BP4_minus_BP1"] = ob_BP4_minus_ob_BP1
    df_ob_price_diffs["BP5_minus_BP1"] = ob_BP5_minus_ob_BP1
    df_ob_price_diffs["BP6_minus_BP1"] = ob_BP6_minus_ob_BP1
    df_ob_price_diffs["BP7_minus_BP1"] = ob_BP7_minus_ob_BP1
    df_ob_price_diffs["BP8_minus_BP1"] = ob_BP8_minus_ob_BP1
    df_ob_price_diffs["BP9_minus_BP1"] = ob_BP9_minus_ob_BP1
    df_ob_price_diffs["BP10_minus_BP1"] = ob_BP10_minus_ob_BP1

    ob_BP1_minus_ob_AP1 = ob_BP1 - ob_AP1
    ob_BP2_minus_ob_AP1 = ob_BP2 - ob_AP1
    ob_BP3_minus_ob_AP1 = ob_BP3 - ob_AP1
    ob_BP4_minus_ob_AP1 = ob_BP4 - ob_AP1
    ob_BP5_minus_ob_AP1 = ob_BP5 - ob_AP1
    ob_BP6_minus_ob_AP1 = ob_BP6 - ob_AP1
    ob_BP7_minus_ob_AP1 = ob_BP7 - ob_AP1
    ob_BP8_minus_ob_AP1 = ob_BP8 - ob_AP1
    ob_BP9_minus_ob_AP1 = ob_BP9 - ob_AP1
    ob_BP10_minus_ob_AP1 = ob_BP10 - ob_AP1

    df_ob_price_diffs["BP1_minus_AP1"] = ob_BP1_minus_ob_AP1
    df_ob_price_diffs["BP2_minus_AP1"] = ob_BP2_minus_ob_AP1
    df_ob_price_diffs["BP3_minus_AP1"] = ob_BP3_minus_ob_AP1
    df_ob_price_diffs["BP4_minus_AP1"] = ob_BP4_minus_ob_AP1
    df_ob_price_diffs["BP5_minus_AP1"] = ob_BP5_minus_ob_AP1
    df_ob_price_diffs["BP6_minus_AP1"] = ob_BP6_minus_ob_AP1
    df_ob_price_diffs["BP7_minus_AP1"] = ob_BP7_minus_ob_AP1
    df_ob_price_diffs["BP8_minus_AP1"] = ob_BP8_minus_ob_AP1
    df_ob_price_diffs["BP9_minus_AP1"] = ob_BP9_minus_ob_AP1
    df_ob_price_diffs["BP10_minus_AP1"] = ob_BP10_minus_ob_AP1

    return df_ob_price_diffs
    
def generate_orderbook_cum_size_log_cum_size(df_ob : pd.DataFrame, 
                                   min_index : int = 0,
                                   max_index : int = -1
                                  ) -> tuple: #(pd.DataFrame, pd.DataFrame):
                                  
    df_ob_cum_size = pd.DataFrame() # empty dataframe
    df_ob_log_cum_size = pd.DataFrame() # empty dataframe
 
    ob_cum_AskSize_1_1 = df_ob["AskSize1"] [min_index : max_index]
    ob_cum_AskSize_1_2 = ob_cum_AskSize_1_1 + df_ob["AskSize2"][min_index : max_index]
    ob_cum_AskSize_1_3 = ob_cum_AskSize_1_2 + df_ob["AskSize3"][min_index : max_index]
    ob_cum_AskSize_1_4 = ob_cum_AskSize_1_3 + df_ob["AskSize4"][min_index : max_index]
    ob_cum_AskSize_1_5 = ob_cum_AskSize_1_4 + df_ob["AskSize5"][min_index : max_index]
    ob_cum_AskSize_1_6 = ob_cum_AskSize_1_5 + df_ob["AskSize6"][min_index : max_index]
    ob_cum_AskSize_1_7 = ob_cum_AskSize_1_6 + df_ob["AskSize7"][min_index : max_index]
    ob_cum_AskSize_1_8 = ob_cum_AskSize_1_7 + df_ob["AskSize8"][min_index : max_index]
    ob_cum_AskSize_1_9 = ob_cum_AskSize_1_8 + df_ob["AskSize9"][min_index : max_index]
    ob_cum_AskSize_1_10 = ob_cum_AskSize_1_9 + df_ob["AskSize10"][min_index : max_index]
    
    df_ob_cum_size["cum_AskSize_1_1"] = ob_cum_AskSize_1_1
    df_ob_cum_size["cum_AskSize_1_2"] = ob_cum_AskSize_1_2
    df_ob_cum_size["cum_AskSize_1_3"] = ob_cum_AskSize_1_3
    df_ob_cum_size["cum_AskSize_1_4"] = ob_cum_AskSize_1_4
    df_ob_cum_size["cum_AskSize_1_5"] = ob_cum_AskSize_1_5
    df_ob_cum_size["cum_AskSize_1_6"] = ob_cum_AskSize_1_6
    df_ob_cum_size["cum_AskSize_1_7"] = ob_cum_AskSize_1_7
    df_ob_cum_size["cum_AskSize_1_8"] = ob_cum_AskSize_1_8
    df_ob_cum_size["cum_AskSize_1_9"] = ob_cum_AskSize_1_9
    df_ob_cum_size["cum_AskSize_1_10"] = ob_cum_AskSize_1_10

    eps = 10**(-6)

    df_ob_log_cum_size["log_cum_AskSize_1_1"] = math.log(math.max(eps, ob_cum_AskSize_1_1))
    df_ob_log_cum_size["log_cum_AskSize_1_2"] = math.log(math.max(eps, ob_cum_AskSize_1_2))
    df_ob_log_cum_size["log_cum_AskSize_1_3"] = math.log(math.max(eps, ob_cum_AskSize_1_3))
    df_ob_log_cum_size["log_cum_AskSize_1_4"] = math.log(math.max(eps, ob_cum_AskSize_1_4))
    df_ob_log_cum_size["log_cum_AskSize_1_5"] = math.log(math.max(eps, ob_cum_AskSize_1_5))
    df_ob_log_cum_size["log_cum_AskSize_1_6"] = math.log(math.max(eps, ob_cum_AskSize_1_6))
    df_ob_log_cum_size["log_cum_AskSize_1_7"] = math.log(math.max(eps, ob_cum_AskSize_1_7))
    df_ob_log_cum_size["log_cum_AskSize_1_8"] = math.log(math.max(eps, ob_cum_AskSize_1_8))
    df_ob_log_cum_size["log_cum_AskSize_1_9"] = math.log(math.max(eps, ob_cum_AskSize_1_9))
    df_ob_log_cum_size["log_cum_AskSize_1_10"] = math.log(math.max(eps, ob_cum_AskSize_1_10))

    ob_cum_BidSize_1_1 = df_ob["BidSize1"][min_index : max_index]
    ob_cum_BidSize_1_2 = ob_cum_BidSize_1_1 + df_ob["BidSize2"][min_index : max_index]
    ob_cum_BidSize_1_3 = ob_cum_BidSize_1_2 + df_ob["BidSize3"][min_index : max_index]
    ob_cum_BidSize_1_4 = ob_cum_BidSize_1_3 + df_ob["BidSize4"][min_index : max_index]
    ob_cum_BidSize_1_5 = ob_cum_BidSize_1_4 + df_ob["BidSize5"][min_index : max_index]
    ob_cum_BidSize_1_6 = ob_cum_BidSize_1_5 + df_ob["BidSize6"][min_index : max_index]
    ob_cum_BidSize_1_7 = ob_cum_BidSize_1_6 + df_ob["BidSize7"][min_index : max_index]
    ob_cum_BidSize_1_8 = ob_cum_BidSize_1_7 + df_ob["BidSize8"][min_index : max_index]
    ob_cum_BidSize_1_9 = ob_cum_BidSize_1_8 + df_ob["BidSize9"][min_index : max_index]
    ob_cum_BidSize_1_10 = ob_cum_BidSize_1_9 + df_ob["BidSize10"][min_index : max_index]
    
    df_ob_cum_size["cum_BidSize_1_1"] = ob_cum_BidSize_1_1
    df_ob_cum_size["cum_BidSize_1_2"] = ob_cum_BidSize_1_2
    df_ob_cum_size["cum_BidSize_1_3"] = ob_cum_BidSize_1_3
    df_ob_cum_size["cum_BidSize_1_4"] = ob_cum_BidSize_1_4
    df_ob_cum_size["cum_BidSize_1_5"] = ob_cum_BidSize_1_5
    df_ob_cum_size["cum_BidSize_1_6"] = ob_cum_BidSize_1_6
    df_ob_cum_size["cum_BidSize_1_7"] = ob_cum_BidSize_1_7
    df_ob_cum_size["cum_BidSize_1_8"] = ob_cum_BidSize_1_8
    df_ob_cum_size["cum_BidSize_1_9"] = ob_cum_BidSize_1_9
    df_ob_cum_size["cum_BidSize_1_10"] = ob_cum_BidSize_1_10

    df_ob_log_cum_size["log_cum_BidSize_1_1"] = math.log(math.max(eps, ob_cum_BidSize_1_1))
    df_ob_log_cum_size["log_cum_BidSize_1_2"] = math.log(math.max(eps, ob_cum_BidSize_1_2))
    df_ob_log_cum_size["log_cum_BidSize_1_3"] = math.log(math.max(eps, ob_cum_BidSize_1_3))
    df_ob_log_cum_size["log_cum_BidSize_1_4"] = math.log(math.max(eps, ob_cum_BidSize_1_4))
    df_ob_log_cum_size["log_cum_BidSize_1_5"] = math.log(math.max(eps, ob_cum_BidSize_1_5))
    df_ob_log_cum_size["log_cum_BidSize_1_6"] = math.log(math.max(eps, ob_cum_BidSize_1_6))
    df_ob_log_cum_size["log_cum_BidSize_1_7"] = math.log(math.max(eps, ob_cum_BidSize_1_7))
    df_ob_log_cum_size["log_cum_BidSize_1_8"] = math.log(math.max(eps, ob_cum_BidSize_1_8))
    df_ob_log_cum_size["log_cum_BidSize_1_9"] = math.log(math.max(eps, ob_cum_BidSize_1_9))
    df_ob_log_cum_size["log_cum_BidSize_1_10"] = math.log(math.max(eps, ob_cum_BidSize_1_10))

    return df_ob_cum_size, df_ob_log_cum_size

