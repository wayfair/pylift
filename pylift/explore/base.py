import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from ..style import _plot_defaults

def _add_bins(df, feats, n_bins=10):
    """Finds n_bins bins of equal size for each feature in dataframe and outputs the result as a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with features
    feats : list
        list of features you would like to consider for splitting into bins (the ones you want to evaluate NWOE, NIV etc for)
    n_bins = number of even sized (no. of data points) bins to use for each feature (this is chosen based on both t and c datasets)

    Returns
    ----------
    df_new : pandas.DataFrame
         original dataframe with bin intervals for each feature included as new columns (labelled as original column name + '_bin')
    """

    df_new = df.copy()

    for feat in feats:
        # check number of unique values of feature -- if low (close to the number of bins), we need to be careful
        num_unique_elements = len(df[feat].unique())

        # we should be more careful with how we make bins
        # we really want to make this independent of bins
        if num_unique_elements > n_bins*2: # x2 because we need intervals
            bin_intervals = pd.qcut(df[feat],n_bins,duplicates='drop') # !!! make sure there's nothing funny happening with duplicates
            # include bins in new column
            df_new[str(feat)+'_bin'] = bin_intervals
        else:
            df_new[str(feat)+'_bin'] = df_new[feat]

    return df_new

def _conv_dict_to_array(a_dict, feats):
    """Converts a_dict to an array

    """
    return [a_dict(feat) for feat in feats]

def _get_counts(df_new, feats, col_treatment='Treatment', col_outcome='Outcome'):
    """Gets all of the counts across the intervals as a dictionary for each feature

    Parameters
    ----------
    df_new : pandas.DataFrame
        the original dataframe of data df with bins included using the _add_bins function
    feats: list
        list of feats to consider

    Returns
    -------
    counts_dict : dictionary
        a dictionary of counts used in other functions to calculate NWOE etc. with keys = feature names and values = dataframe of counts

    """

    counts_dict = {}
    y = df_new[col_treatment]
    trt = df_new[col_outcome]
    for feat in feats:
        bin_feat = str(feat)+'_bin'
        counts1_t1 = df_new[(y==1)&(trt==1)][[feat,bin_feat]].groupby(bin_feat).count().rename(columns={feat:'counts_y1t1'})
        counts1_t0 = df_new[(y==1)&(trt==0)][[feat,bin_feat]].groupby(bin_feat).count().rename(columns={feat:'counts_y1t0'})
        counts0_t1 = df_new[(y==0)&(trt==1)][[feat,bin_feat]].groupby(bin_feat).count().rename(columns={feat:'counts_y0t1'})
        counts0_t0 = df_new[(y==0)&(trt==0)][[feat,bin_feat]].groupby(bin_feat).count().rename(columns={feat:'counts_y0t0'})
        # creating a dataframe with all of these results
        counts_dict[feat] = pd.concat([counts1_t1,counts1_t0,counts0_t1,counts0_t0],axis=1).fillna(0)+1 # replace any empty slots with zeros (and add 1 to everything)

    return counts_dict

def _WOE(df_new, feats, trt, col_treatment='Treatment', col_outcome='Outcome'):
    """The WOE for our dataset (with trt telling us which treatment to consider)

    Parameters
    ----------
    df_new : pandas.DataFrame
        original dataframe with bin intervals for each feature included as new columns
    feats : list
        features of interest
    trt : int
        0 or 1 indicating treatment.

    Returns
    -------
    WOE
        Weight of evidence for each feature by bin (a dictionary with keys = feature names)
    """

    # get the counts for each y & trt pair
    counts_dict = _get_counts(df_new, feats, col_treatment=col_treatment, col_outcome=col_outcome)

    # create dictionary WOE of each feature
    WOE = {}

    for feat in feats:
        # calculate the WOE for each bin
        if trt==1:
            WOE_indiv = pd.DataFrame({'WOE':np.log(counts_dict[feat]['counts_y1t1']/counts_dict[feat]['counts_y0t1']*sum(counts_dict[feat]['counts_y0t1'])/sum(counts_dict[feat]['counts_y1t1']))})
        elif trt==0:
            WOE_indiv = pd.DataFrame({'WOE':np.log(counts_dict[feat]['counts_y1t0']/counts_dict[feat]['counts_y0t0']*sum(counts_dict[feat]['counts_y0t0'])/sum(counts_dict[feat]['counts_y1t0']))})
        # could also add a general case for no trt value -- add both t0 t1 columns

        WOE[feat] = WOE_indiv

    return WOE

def _NWOE(df_new, feats, col_treatment='Treatment', col_outcome='Outcome'):
    """
    Net Weight of Evidence by feature and bin

    Parameters
    ----------
    df_new : pandas.DataFrame
        original dataframe with bin intervals for each feature included as new columns
    feats : list
        features of interest

    Returns
    -------
    NWOE :
        the Net Weight of Evidence (weighted by density as each interval can
        hold different amounts of data -- this is more meaningful) for each
        feature by interval (a dictionary with keys = feature names)

    """

    # first get the bins -- we want to use the same bins for trt and contr
    # using both sets of data together should help get the intervals more accurately

    # getting the counts again
    counts_dict = _get_counts(df_new, feats, col_treatment=col_treatment, col_outcome=col_outcome)

    # get the WOEs for trt=0 and trt=1
    WOEs_trt1 = _WOE(df_new, feats, trt=1, col_treatment=col_treatment, col_outcome=col_outcome)
    WOEs_trt0 = _WOE(df_new, feats, trt=0, col_treatment=col_treatment, col_outcome=col_outcome)

    # combine into NWOE dictionary
    NWOE = {}
    for feat in feats:
        #NWOE[feat] = WOE_df['WOE_trt1'] - WOE_df['WOE_trt0']
        # include density -- get counts
        counts = counts_dict[feat]['counts_y0t0']+counts_dict[feat]['counts_y0t1']+counts_dict[feat]['counts_y1t0']+counts_dict[feat]['counts_y1t1']

        NWOE[feat] = pd.DataFrame({'NWOE':(WOEs_trt1[feat] - WOEs_trt0[feat])['WOE']*counts/sum(counts)})

    return NWOE

# note that with the modified NWOE we now already include the density -- so should be removed
def _NIV(df_new, feats, col_treatment='Treatment', col_outcome='Outcome'):
    """Net Information Value by feature

    Parameters
    ----------
    - df_new = original dataframe with bin intervals for each feature included as new columns
    - feats = features of interest

    Returns
    -------
    - NIV = Net Information Value for each feature (a dictionary with keys = feature names)

    """
    NWOE_dict = _NWOE(df_new, feats, col_treatment=col_treatment, col_outcome=col_outcome)

    # calculating the normalization for probabilities
    # this requires overall count of number of people with y=0,1 in trt=0,1
    trt = df_new[col_treatment]
    y = df_new[col_outcome]
    ny0t0 = len(df_new[(y==0)&(trt==0)])
    ny0t1 = len(df_new[(y==0)&(trt==1)])
    ny1t0 = len(df_new[(y==1)&(trt==0)])
    ny1t1 = len(df_new[(y==1)&(trt==1)])

    # get overall counts
    counts_dict = _get_counts(df_new, feats, col_treatment=col_treatment, col_outcome=col_outcome)

    NIV = {}
    for feat in feats:
        # get the counts for y=1,y=0 & trt=1,trt=0 for each feature by bin
        # combine into front term
        NIV_weight = (counts_dict[feat]['counts_y1t1']*counts_dict[feat]['counts_y0t0']/(sum(counts_dict[feat]['counts_y1t1'])*sum(counts_dict[feat]['counts_y0t0'])) - counts_dict[feat]['counts_y1t0']*counts_dict[feat]['counts_y0t1']/(sum(counts_dict[feat]['counts_y1t0'])*sum(counts_dict[feat]['counts_y0t1'])))
        # get NWOE & combine in one df
        NIV_feat = 100*(NIV_weight*NWOE_dict[feat]['NWOE']).sum() # already included the density term in NWOE!
        # We don't need to x 100, but it makes the numbers closer to 1 -- just a convention

        NIV[feat] = NIV_feat

    return NIV

def _NIV_bootstrap(df, feats, n_bins=10, perc=[20,80], n_iter=100, frac=0.5, col_treatment='Treatment', col_outcome='Outcome'):
    """
    Calculates the NIV for each using bootstrapped samples with means, lower and upper percentiles
    to be used for determine which features to use.

    Parameters
    ----------
    - df = the training dataframe with labels 'y', treatment label 'trt', features columns names 'feats' without bins included
    - feats = features of interest
    - perc = list with upper and lower percentiles to calculate from the bootstrapped NIV
    - n_iter = number of bootstraps to take
    - frac = percentage of samples to use for each bootstrap

    Returns
    -------
    - means_dict, low_perc_dict, high_perc_dict = dictionary of means, low percentile, high percentile for each feature across the boostrapped samples
    """

    # array of NIV dictionaries (one for each iteration)
    NIV_dict_array = []

    for i in np.arange(n_iter):
        # including bins for each subset of the dataset
        df_sub_bins = _add_bins(df.sample(frac=frac), feats, n_bins=n_bins)
        # finding the NIV for this subset
        NIV_dict = _NIV(df_sub_bins,feats, col_treatment=col_treatment, col_outcome=col_outcome)
        NIV_dict_array.append(NIV_dict)

    # replacing with a dictionary of arrays
    NIV_array_dict = {}
    for feat in feats:
        NIV_array_dict[feat] = []
        for i in np.arange(len(NIV_dict_array)):
            NIV_array_dict[feat].append(NIV_dict_array[i][feat])

    # creating dictionary of means, lower percentile, upper percentile
    means_dict = {}
    low_perc_dict = {}
    high_perc_dict = {}

    for feat in feats:
        bs_array = np.array(NIV_array_dict[feat])
        # replace any infs with 0
        bs_array[bs_array==np.inf]=0
        # mean values
        means_dict[feat] = np.mean(bs_array)
        low_perc_dict[feat] = np.percentile(bs_array,perc[0])
        high_perc_dict[feat] = np.percentile(bs_array,perc[1])

    return means_dict, low_perc_dict, high_perc_dict

def _plot_NWOE_bins(NWOE_dict, feats):
    """
    Plots the NWOE by bin for the subset of features interested in (form of list)

    Parameters
    ----------
    - NWOE_dict = dictionary output of `NWOE` function
    - feats = list of features to plot NWOE for

    Returns
    -------
    - plots of NWOE for each feature by bin
    """

    for feat in feats:
        fig, ax = _plot_defaults()
        feat_df = NWOE_dict[feat].reset_index()
        plt.bar(range(len(feat_df)), feat_df['NWOE'], tick_label=feat_df[str(feat)+'_bin'], color='k', alpha=0.5)
        plt.xticks(rotation='vertical')
        ax.set_title('NWOE by bin for '+str(feat))
        ax.set_xlabel('Bin Interval');
    return ax

def _plot_NIV_bs(means_dict, low_perc_dict, high_perc_dict, feats):
    """
    Plots the NWOE by bin for the subset of features interested in (form of list)

    Parameters
    ----------
    - NWOE_dict = dictionary output of `NWOE` function
    - feats = list of features to plot NWOE for

    Returns
    -------
    - plots of NWOE for each feature by bin
    """

    # find order of features from highest mean value to lowest
    # could also order by low_perc_dict
    feats_sorted = sorted(means_dict, key=means_dict.get, reverse=False)

    # convert to arrays
    means = np.array([means_dict[feat] for feat in feats_sorted])
    low_perc = np.array([low_perc_dict[feat] for feat in feats_sorted])
    high_perc = np.array([high_perc_dict[feat] for feat in feats_sorted])

    ind = np.arange(len(feats))  # the x locations for the feats
    fig, ax = _plot_defaults(figsize=(15,len(feats)))
    ax.barh(ind, means, xerr=[means-low_perc,high_perc-means], align='center', alpha=0.5, ecolor='black', capsize=8)
    ax.set_ylim([-0.75,len(feats)-0.25])
    ax.set_yticks(ind)
    ax.set_yticklabels(feats_sorted, minor=False)
    ax.set_ylabel('Features')
    ax.set_xlabel('NIV')
    return ax
