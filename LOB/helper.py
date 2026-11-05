import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def this_sqrt(x):
    return math.sqrt(max(x,0.0))

# Apply the function using np.vectorize
this_vec_sqrt = np.vectorize(this_sqrt)

def this_log(x):
    eps = 10**(-8)
    return math.log(max(x,eps))

# Apply the function using np.vectorize
this_vec_log = np.vectorize(this_log)

def expw_zscore(df : pd.DataFrame, 
                lookback : int = 501,
                include_time_column : bool = False
                ) -> pd.DataFrame:
                    
    eps = 10**(-6)

    lookback_decay_value = 0.01 # lambda**lookback = lookback_decay_value
    this_lambda = math.exp( math.log(lookback_decay_value) / lookback)
    this_alpha = 1.0 - this_lambda
    
    df_output = pd.DataFrame() # empty dataframe

    these_columns = list(df.columns).copy()
    for col_name in these_columns:
        if col_name == "Time":
            continue
        mu_col = col_name + "_mu"
        np_mu_col = df[col_name].ewm(alpha=this_alpha, adjust=False).mean().shift(1)
        np_mu_col[0] = df[col_name].iloc[0]
        
        sq_dev = (df[col_name] - np_mu_col) ** 2
        np_V_col = sq_dev.ewm(alpha=this_alpha, adjust=False).mean().shift(1)
        np_V_col[0] = 0.0
        
        np_sigma_col = this_vec_sqrt(np_V_col + eps)

        z_col = col_name + "_z"
        np_z_col = (df[col_name] - np_mu_col) / np_sigma_col

        df_output[z_col] = np_z_col[lookback:]
      
    # put the time column back in, if it was initially present
    if include_time_column & ("Time" in df.columns):
        df_output["Time"] = df["Time"][lookback:]
        # now rearrange so that time is the very first column
        this_list = ["Time"]
        df_output = df_output[this_list + [c for c in df_output if c not in this_list] ]

    return df_output
    
def read_and_format_orderbook(orderbook_file : str = "UNDEFINED",
                                min_index : int = -1,
                                max_index : int = -1,
                                orderbook_depth : int = 10) -> pd.DataFrame:
    
    df_ob = pd.DataFrame() # empty dataframe
    if min_index > 0 and max_index > min_index:
        df_ob = pd.read_csv(orderbook_file, skiprows=range(1, min_index + 1), nrows = max_index - min_index)
    else:
        df_ob=pd.read_csv(orderbook_file)
        
    # remove the unwanted columns
    df_ob = df_ob.iloc[:, 0:4 * orderbook_depth]
        
    # Add column names
    ap = "AskPrice"
    a_s = "AskSize"
    bp = "BidPrice"
    b_s = "BidSize"
    col_names = [] 
    for i in range(1,orderbook_depth+1):
        col_names.append(ap + str(i))
        col_names.append(a_s + str(i))
        col_names.append(bp + str(i))
        col_names.append(b_s + str(i))
    df_ob.columns= col_names
    
    return df_ob
    
def read_and_format_messages(message_file : str = "UNDEFINED",
                                min_index : int = -1,
                                max_index : int = -1
                                ) -> pd.DataFrame:
    
    df_m = pd.DataFrame() # empty dataframe
    if min_index > 0 and max_index > min_index:
        df_m = pd.read_csv(message_file, skiprows=range(1, min_index + 1), nrows = max_index - min_index)
    else:
        df_m=pd.read_csv(message_file)
             
    # Add column names
    df_m.columns = ["Time", "EventType", "OrderID", "Size", "Price", "Direction", "Unknown"]
    
    return df_m
    
def generate_orderbook_price_diffs(df_ob_price_levels : pd.DataFrame, 
                                   min_index : int = 0,
                                   max_index : int = -1,
                                   orderbook_depth : int = 10,
                                   include_time_column : bool = False
                                  ) -> pd.DataFrame:
                                  
    min_orderbook_depth = 1
    max_orderbook_depth = 10
    orderbook_depth = max(min_orderbook_depth, min(max_orderbook_depth, orderbook_depth))
    
    num_rows = max_index - min_index - 1
    if max_index == -1:
        num_rows = df_ob_price_levels.shape[0] - min_index - 1
    
    # define and populate ob_AP and ob_BP
    ob_AP = np.zeros((orderbook_depth + 1, num_rows))
    ob_BP = np.zeros((orderbook_depth + 1, num_rows))
    
    # column zero of ob_AP and ob_BP will be empty, this is intentional
    for i in range(1, orderbook_depth + 1):
        ob_AP[i,:] = df_ob_price_levels['AskPrice' + str(i)] [min_index : max_index]
        ob_BP[i,:] = df_ob_price_levels['BidPrice' + str(i)] [min_index : max_index]

    # define and populate df_ob_price_diffs
    df_ob_price_diffs = pd.DataFrame() # empty dataframe
 
    # price diffs : AP - AP1, BP - BP1
    for i in range(2, orderbook_depth + 1):
        df_ob_price_diffs["AP" + str(i) + "_minus_AP1"] = ob_AP[i,:] - ob_AP[1,:]
        df_ob_price_diffs["BP" + str(i) + "_minus_BP1"] = ob_BP[i,:] - ob_BP[1,:]
        
    # price diffs : AP - BP1, BP - AP1
    for i in range(1, orderbook_depth + 1):
        df_ob_price_diffs["AP" + str(i) + "_minus_BP1"] = ob_AP[i,:] - ob_BP[1,:]
        df_ob_price_diffs["BP" + str(i) + "_minus_AP1"] = ob_BP[i,:] - ob_AP[1,:]
        
    # put the time column back in
    if include_time_column:
        df_ob_price_diffs["Time"] = df_ob_price_levels["Time"]

    return df_ob_price_diffs
    

def generate_orderbook_cum_size_log_cum_size(df_ob : pd.DataFrame, 
                                   min_index : int = 0,
                                   max_index : int = -1,
                                   orderbook_depth : int = 10,
                                   include_time_column : bool = False
                                  ) -> tuple: 
                                  
    df_ob_cum_size = pd.DataFrame() # empty dataframe
    df_ob_log_cum_size = pd.DataFrame() # empty dataframe
    df_ob_imbalance = pd.DataFrame() # empty dataframe
    df_ob_log_imbalance = pd.DataFrame() # empty dataframe
    df_ob_total_size = pd.DataFrame() # empty dataframe
    df_ob_log_total_size = pd.DataFrame() # empty dataframe
    
    # cumulative size
    df_ob_cum_size["cum_AskSize_1_1"] = df_ob["AskSize1"] [min_index : max_index]
    df_ob_cum_size["cum_BidSize_1_1"] = df_ob["BidSize1"] [min_index : max_index]
    for i in range(2, orderbook_depth + 1):
        as_key = "cum_AskSize_1_" + str(i)
        bs_key = "cum_BidSize_1_" + str(i)
        df_ob_cum_size[as_key] = df_ob_cum_size["cum_AskSize_1_" + str(i-1)] + df_ob["AskSize" + str(i)][min_index : max_index]
        df_ob_cum_size[bs_key] = df_ob_cum_size["cum_BidSize_1_" + str(i-1)] + df_ob["BidSize" + str(i)][min_index : max_index]

    # log cumulative size
    for i in range(1, orderbook_depth + 1):
        as_key = "cum_AskSize_1_" + str(i)
        bs_key = "cum_BidSize_1_" + str(i)
        df_ob_log_cum_size["log_" + as_key] = this_vec_log(df_ob_cum_size[as_key])
        df_ob_log_cum_size["log_" + bs_key] = this_vec_log(df_ob_cum_size[bs_key])
    
    num_rows = df_ob_cum_size.shape[0]
    eps = 10**(-6)
    eps_vec = eps * np.ones(num_rows)
    
    # imbalance and log imbalance (imbalance of logs)
    for i in range(1, orderbook_depth + 1):
        bma_key = "cum_BidSize_minus_AskSize_1_" + str(i)
        as_key = "cum_AskSize_1_" + str(i)
        bs_key = "cum_BidSize_1_" + str(i)
        df_ob_imbalance[bma_key] = (df_ob_cum_size[bs_key] - df_ob_cum_size[as_key]) / (df_ob_cum_size[bs_key] + df_ob_cum_size[as_key] + eps_vec)
        df_ob_log_imbalance[bma_key] = (this_vec_log(df_ob_cum_size[bs_key]) - this_vec_log(df_ob_cum_size[as_key])) / (this_vec_log(df_ob_cum_size[bs_key]) + this_vec_log(df_ob_cum_size[as_key] + eps_vec))
        
    # total cumulative size (Bid + Ask), log total cumulative size
    for i in range(1, orderbook_depth + 1):
        apb_key = "cum_BidSize_plus_AskSize_1_" + str(i)
        as_key = "cum_AskSize_1_" + str(i)
        bs_key = "cum_BidSize_1_" + str(i) 
        df_ob_total_size[apb_key] = df_ob_cum_size[bs_key] + df_ob_cum_size[as_key]
        df_ob_log_total_size[apb_key] = this_vec_log(df_ob_total_size[apb_key])
       
    # put the time column back in
    if include_time_column:
        df_ob_cum_size["Time"] = df_ob["Time"] 
        df_ob_log_cum_size["Time"] = df_ob["Time"]
        df_ob_imbalance["Time"] = df_ob["Time"]
        df_ob_log_imbalance["Time"] = df_ob["Time"]
        df_ob_total_size["Time"] = df_ob["Time"]
        df_ob_log_total_size["Time"] = df_ob["Time"]

    return df_ob_cum_size, df_ob_log_cum_size, df_ob_imbalance, df_ob_log_imbalance, df_ob_total_size, df_ob_log_total_size

def create_representation_one(ob_file_name : str = "UNDEFINED",
        m_file_name : str = "UNDEFINED",
        use_price_diffs_z_scores : bool = True,
        use_cum_size_z_scores : bool = True,
        use_log_cum_size_z_scores : bool = True,
        use_ob_imbalance : bool = True,
        use_ob_imbalance_z_scores : bool = True,
        use_ob_log_imbalance : bool = True,
        use_ob_log_imbalance_z_scores : bool = True,
        use_total_size_z_scores : bool = True,
        use_log_total_size_z_scores : bool = True,
        lookback : int = 501,
        min_index : int = 0, 
        max_index : int = -1,
        orderbook_depth : int = 10
        ) -> pd.DataFrame:
        
        df_ob = read_and_format_orderbook(ob_file_name, min_index, max_index, orderbook_depth)
        df_m = read_and_format_messages(m_file_name, min_index, max_index)
        df_ob["Time"] = df_m["Time"]
        
        df_price_diffs = pd.DataFrame()
        df_price_diffs_z_scores = pd.DataFrame()
        df_ob_cum_size = pd.DataFrame() 
        df_ob_cum_size_z_scores = pd.DataFrame()
        df_ob_log_cum_size = pd.DataFrame() 
        df_ob_log_cum_size_z_scores = pd.DataFrame() 
        df_ob_imbalance = pd.DataFrame()
        df_ob_imbalance_z_scores = pd.DataFrame() 
        df_ob_log_imbalance = pd.DataFrame()        
        df_ob_log_imbalance_z_scores = pd.DataFrame()
        df_ob_total_size_z_scores = pd.DataFrame()
        df_ob_log_total_size_z_scores = pd.DataFrame()
            
        if use_price_diffs_z_scores:
            df_price_diffs = generate_orderbook_price_diffs(df_ob, orderbook_depth = orderbook_depth)
            df_price_diffs_z_scores = expw_zscore(df_price_diffs, lookback)
            
        if (use_cum_size_z_scores or use_log_cum_size_z_scores):  
            df_ob_cum_size, df_ob_log_cum_size, df_ob_imbalance, df_ob_log_imbalance, df_ob_total_size, df_ob_log_total_size = generate_orderbook_cum_size_log_cum_size(df_ob, orderbook_depth = orderbook_depth)
            df_ob_cum_size_z_scores = expw_zscore(df_ob_cum_size, lookback)
            df_ob_log_cum_size_z_scores = expw_zscore(df_ob_log_cum_size, lookback)
            
        if (use_ob_imbalance_z_scores or use_ob_log_imbalance_z_scores):    
            df_ob_imbalance_z_scores = expw_zscore(df_ob_imbalance, lookback)
            df_ob_log_imbalance_z_scores = expw_zscore(df_ob_log_imbalance, lookback)
            
        if (use_total_size_z_scores or use_log_total_size_z_scores):
            df_ob_total_size_z_scores = expw_zscore(df_ob_total_size, lookback) 
            df_ob_log_total_size_z_scores = expw_zscore(df_ob_log_total_size, lookback)
        
        df_representation_one = pd.DataFrame() # empty dataframe
        if use_price_diffs_z_scores:
            if df_representation_one.size == 0:
                df_representation_one = df_price_diffs_z_scores
            else:
                df_representation_one = df_representation_one.join(df_price_diffs_z_scores)
        if use_cum_size_z_scores:
            if df_representation_one.size == 0:
                df_representation_one = df_ob_cum_size_z_scores
            else:
                df_representation_one = pd.concat([df_representation_one, df_ob_cum_size_z_scores], axis=1)
        if use_log_cum_size_z_scores:
            if df_representation_one.size == 0:
                df_representation_one = df_ob_log_cum_size_z_scores
            else:
                df_representation_one = pd.concat([df_representation_one, df_ob_log_cum_size_z_scores], axis=1)
        if use_ob_imbalance:
            if df_representation_one.size == 0:
                df_representation_one = df_ob_imbalance[lookback:]
            else:
                df_representation_one = pd.concat([df_representation_one, df_ob_imbalance[lookback:]], axis=1)
        if use_ob_imbalance_z_scores:
            if df_representation_one.size == 0:
                df_representation_one = df_ob_imbalance_z_scores
            else:
                df_representation_one = pd.concat([df_representation_one, df_ob_imbalance_z_scores], axis=1)
        if use_ob_log_imbalance:
            if df_representation_one.size == 0:
                df_representation_one = df_ob_log_imbalance[lookback:]
            else:
                df_representation_one = pd.concat([df_representation_one, df_ob_log_imbalance[lookback:]], axis=1)
        if use_ob_log_imbalance_z_scores:
            if df_representation_one.size == 0:
                df_representation_one = df_ob_log_imbalance_z_scores
            else:
                df_representation_one = pd.concat([df_representation_one, df_ob_log_imbalance_z_scores], axis=1)
        if use_total_size_z_scores:
            if df_representation_one.size == 0:
                df_representation_one = df_ob_total_size_z_scores
            else:
                df_representation_one = pd.concat([df_representation_one, df_ob_total_size_z_scores], axis=1)
        if use_total_size_z_scores:
            if df_representation_one.size == 0:
                df_representation_one = df_ob_log_total_size_z_scores
            else:
                df_representation_one = pd.concat([df_representation_one, df_ob_log_total_size_z_scores], axis = 1)
                
        # put the time column back in
        df_representation_one["Time"] = df_ob["Time"][lookback:]
        # now rearrange so that time is the very first column
        this_list = ["Time"]
        df_representation_one = df_representation_one[this_list + [c for c in df_representation_one if c not in this_list] ]
        
        return df_representation_one
