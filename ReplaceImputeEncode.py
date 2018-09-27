#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 04:51:24 2018

@author: Edward Jones

@parameters:
    *** __init__() ***
    data_map - The metadata dictionary.  If not passed, a metadata
        dictionary is created from the data. Each column in the data must
        be described in the metadata.  The dictionary keys correspond to
        the names for each column.  The value for each key is a list of
        three objects:
            1.  An integer indicating data type - 0=interval, 1=binary, 
                2=nominal, and any integer above 2 is to be ignored.
            2.  A tuple containing the lower and upper bounds for integer
                attributes, or a list of allowed categories for binary
                and nominal attributes.  These can be integers or strings.
            3.  A list of two integers calculated by the class.  Initially
                they should be set to zero.  The fit() method of this class
                calculates the number of missing cases and the number of 
                outliers.  They are stored the 3rd object, a list, in 
                these metadata.
                
    nominal_encoding - Can be 'one-hot', 'SAS' or None.
    
    interval_scale   - Can be 'std', 'robust' or None.
    
    drop             - True of False.  True drops the last nominal encoded
                       columns.  False keeps all nominal encoded columns.
    
    display          - True or False.  True displays the number of missing 
                        and outliers found in the data.
                        
    *** fit() & fit_transform () ***
    df  - a pandas DataFrame containing the data description
                        by the metadata found in attributes_map (required)
    attributes_map - See above description.
    
@Cautions:
    The incoming DataFrame, df, and the attributes_map are deepcopied to
    ensure that changes to the DataFrame are only held within the class
    object self.copy_df.  The attributes_map is deepcopied into
    self.features_map.  The categorical values are change to numeric and
    the number of missing and number of outliers are also changed from zero.
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
#from copy import deepcopy

class ReplaceImputeEncode(object):
    def __init__(self, data_map=None, nominal_encoding='SAS', \
                 interval_scale=None, drop=True, display=False):
        if interval_scale=='None' or interval_scale=='none':
            self.interval_scale=None
        else:
            self.interval_scale=interval_scale
        self.go_flag = False
        self.features_map = data_map
        self.drop    = drop
        self.display = display
        self.interval_scale = interval_scale
        if nominal_encoding=='None' or nominal_encoding=='none':
            self.nominal_encoding = None
        else:
            self.nominal_encoding = nominal_encoding
        #nominal_encoding can be 'SAS' of 'one-hot'
        if nominal_encoding != 'SAS' and nominal_encoding != 'one-hot' \
            and nominal_encoding != None:
            print("***Error:  nominal_encoding selection invalid.")
            print("***Must be set to SAS, one-hot, or None")
            return
        if interval_scale != 'std' and interval_scale != 'robust' \
            and interval_scale != None:
            print("***Error:  interval_scale must be 'std', 'robust' or None")
            return
        if data_map==None:
            print("Attributes Map required in fit() method \n")
            return
        #self.features_map = deepcopy(data_map)
        self.features_map = data_map
        self.interval_attributes = []
        self.nominal_attributes  = []
        self.binary_attributes   = []
        self.onehot_attributes   = []
        self.hot_drop_list       = []
        for feature,v in self.features_map.items():
            if v[0]==0:
                self.interval_attributes.append(feature)
            else:
                if v[0]==1:
                    self.binary_attributes.append(feature)
                else:
                    if v[0]>2: # Ignore, don't touch this attributes
                        continue
                    # It's Nominal
                    self.nominal_attributes.append(feature)
                    # Setup column names for encoding, all but the last one
                    n_cat = len(v[1])
                    if self.drop == True:
                        n_cat -= 1
                    for i in range(n_cat):
                        str = feature+("%i" %i)
                        self.onehot_attributes.append(str)
                    #self.hot_drop_list.append(str)

        self.n_interval = len(self.interval_attributes)
        self.n_binary   = len(self.binary_attributes)
        self.n_nominal  = len(self.nominal_attributes)
        self.n_onehot   = len(self.onehot_attributes)
        self.cat        = self.n_binary + self.n_nominal
        self.col = []
        for i in range(self.n_interval):
            self.col.append(self.interval_attributes[i])
        for i in range(self.n_binary):
            self.col.append(self.binary_attributes[i])
        if self.nominal_encoding==None:
            for i in range(self.n_nominal):
                self.col.append(self.nominal_attributes[i])
        else:
            for i in range(self.n_onehot):
                self.col.append(self.onehot_attributes[i])

        self.go_flag = True
    def fit(self, df, data_map=None):
        #self.df_copy = deepcopy(df)
        self.df_copy = df
        if data_map==None and self.features_map==None:
            print("No Attributes Map Given ********.\n")
            print("Using draft_features_map() method to construct one.")
            print("Check the Attributes Map for Data Consistency.")
            self.features_map = \
                    self.draft_features_map(self.df_copy, display_map=True)
        else: 
            if self.features_map == None:
                self.features_map = data_map
                #self.features_map = deepcopy(data_map)

        self.interval_attributes = []
        self.nominal_attributes  = []
        self.binary_attributes   = []
        self.onehot_attributes   = []
        self.hot_drop_list       = []
        for feature,v in self.features_map.items():
            if v[0]==0:
                self.interval_attributes.append(feature)
            else:
                if v[0]==1:
                    self.binary_attributes.append(feature)
                else:
                    if v[0]==2:
                        self.nominal_attributes.append(feature)
                        for i in range(len(v[1])):
                            str = feature+("%i" %i)
                            self.onehot_attributes.append(str)
                        if self.drop==True:
                            self.hot_drop_list.append(str)
                    else:
                        # Ignore this attribute
                        continue
        self.n_interval = len(self.interval_attributes)
        self.n_binary   = len(self.binary_attributes)
        self.n_nominal  = len(self.nominal_attributes)
        self.n_onehot   = len(self.onehot_attributes)
        self.cat        = self.n_binary + self.n_nominal
        self.n_obs  = df.shape[0]
        #self.n_cols = df.shape[1]
        self.col = []
        for i in range(self.n_interval):
            self.col.append(self.interval_attributes[i])
        for i in range(self.n_binary):
            self.col.append(self.binary_attributes[i])
        if self.nominal_encoding==None:
            for i in range(self.n_nominal):
                self.col.append(self.nominal_attributes[i])
        else:
            for i in range(self.n_onehot):
                self.col.append(self.onehot_attributes[i])

        if self.display:
            print("\n********** Data Preprocessing ***********")
            print("Features Dictionary Contains:\n%i Interval," \
                  %self.n_interval, "\n%i Binary, and" %self.n_binary,\
                  "\n%i Nominal Attribute(s).\n"\
                  %self.n_nominal)
            print("Data contains %i observations & %i columns.\n" %df.shape)
        self.initial_missing = df.isnull().sum()
        self.feature_names = np.array(df.columns.values)
        for feature in self.feature_names:
            if self.initial_missing[feature]>(self.n_obs/2):
                print(feature+":\n\t%i missing: Drop this attribute." \
                  %self.initial_missing[feature])
        # Initialize number missing in attribute_map
        for feature,v in self.features_map.items():
            v[2][0] = self.initial_missing[feature]

        # Scan for outliers among interval attributes
        nan_map = df.isnull()
        for i in range(self.n_obs):
        # Check for outliers in interval attributes
            for feature, v in self.features_map.items():
                if nan_map.loc[i,feature]==True:
                    continue
                if v[0]==0: # Interval Attribute
                    l_limit = v[1][0]
                    u_limit = v[1][1]
                    if df.loc[i,feature]>u_limit or df.loc[i,feature]<l_limit:
                        v[2][1] += 1
                        self.df_copy.loc[i,feature] = None
                else: 
                    if v[0]>2: # don't touch this attribute
                        continue
                    # Categorical Attribute
                    in_cat = False
                    for cat in v[1]:
                        if df.loc[i,feature]==cat:
                            in_cat=True
                    if in_cat==False:
                        self.df_copy.loc[i,feature] = None
                        v[2][1] += 1
        if self.display:
            print("\nAttribute Counts")
            max_label = 0
            for k, v in self.features_map.items():
                if len(k) > max_label:
                    max_label = len(k)
            max_label += 2
            label_format = ("{:.<%i" %(max_label+5))+"s}{:>8s}{:>10s}" 
            print(label_format.format('', 'Missing', 'Outliers'))
            label_format = ("{:.<%i" %max_label)+"s}{:10d}{:10d}"
            for k,v in self.features_map.items():
                print(label_format.format(k, v[2][0], v[2][1]))
            
    def draft_features_map(self, df, display_map=True):
        feature_names = np.array(df.columns.values)
        draft_features_map = {}
        for feature in feature_names:
            n = df[feature].value_counts()
            mis = df[feature].isnull().sum()
            if len(n) > 10:
                # Attribute is Interval
                min = df[feature].min()
                max = df[feature].max()
                draft_features_map[feature]=[0,(min, max), [mis,0]]
                continue
            if len(n) == 2: 
                # Attribute is Binary
                val1 = df[feature].unique()[0]
                val2 = df[feature].unique()[1]
                draft_features_map[feature]=[1,(val1, val2), [mis,0]]
            else:
                # Attribure is Nominal or Ordinal
                val = []
                a   = df[feature].unique()
                for i in range(len(n)):
                    val.append(a[i])
                categories = tuple(val)
                draft_features_map[feature]=[2,categories, [mis,0]]
        if display_map:
            # print the features map
            print("************* FEATURE MAP **************\n")
            last_feature = feature_names[-1]
            print("aggregate_map = {")
            for feature,v in draft_features_map.items():
                if feature==last_feature:
                    print("\t'"+feature+"':", v, "}\n")
                else:
                    print("\t'"+feature+"':", v, ",")
                    
            for feature, v in draft_features_map.items():
                if v[0] == 0:
                    print(feature+": Assumed Interval Attribute")
                    print("    with lower bound ", v[1][0], "and", \
                          "upper bound.", v[1][1])
                if v[0] == 1:
                    print(feature+": Assumed Binary Attribute.")
                if v[0] == 2:
                    print(feature+": Assumed Nominal Attribute with", \
                          len(v[1]), "Categories.")
                print("")
        return draft_features_map
    
    def display_features_map(self):
        # print the features map
        print("************* FEATURE MAP **************\n")
        for feature,v in self.features_map.items():
            print("'"+feature+"':", v, "\n")
            
    def impute(self):
        self.impute_interval()
        self.impute_binary()
        self.impute_nominal()
        self.imputed_data()
    
    def impute_interval(self):
        if (self.n_interval==0):
            self.imputed_interval_data = np.empty((self.n_obs, 0))
            return
        # Put the interval data from the dataframe into a numpy array
        interval_data= self.df_copy.as_matrix(columns=self.interval_attributes)
        # Create the Imputer for the Interval Data
        self.interval_imputer = preprocessing.Imputer(strategy='mean')
        # Impute the missing values in the Interval data
        self.imputed_interval_data = \
            self.interval_imputer.fit_transform(interval_data)
    def impute_binary(self):
        if (self.n_binary==0):
            self.imputed_binary_data = np.empty((self.n_obs, 0))
            return
        # Put the nominal and binary data from the dataframe into a numpy array
        #cat_df = df[self.binary_attributes]
        cat_df = pd.DataFrame(columns=self.binary_attributes)
        for feature in self.binary_attributes:
            cat_df[feature]= self.df_copy[feature].astype('category').cat.codes
            cat_df.loc[cat_df[feature]==-1, feature] = None
        cat_array = cat_df.as_matrix()
        # Create Imputer for Categorical Data
        cat_imputer = preprocessing.Imputer(strategy='most_frequent')
        # Impute the missing values in the Categorical Data
        self.imputed_binary_data = \
            cat_imputer.fit_transform(cat_array)
            
    def impute_nominal(self):
        if (self.n_nominal==0):
            self.imputed_nominal_data = np.empty((self.n_obs, 0))
            return
        # Put the nominal and binary data from the dataframe into a numpy array
        cat_df  = pd.DataFrame(columns=self.nominal_attributes)
        for feature in self.nominal_attributes:
            cat_df[feature]= self.df_copy[feature].astype('category').cat.codes
            cat_df.loc[cat_df[feature]==-1, feature] = None
        cat_array = cat_df.as_matrix()
        # Create Imputer for Categorical Data
        cat_imputer = preprocessing.Imputer(strategy='most_frequent')
        # Impute the missing values in the Categorical Data
        self.imputed_nominal_data = \
            cat_imputer.fit_transform(cat_array)
            
    def imputed_data(self):
        # Bring Interval and Categorial Data Together into a dataframe
        # The Imputed Data
        # Col is not the same as self.col.  col contains the main attribute
        # names, self.col contains the one-hot names
        col = self.interval_attributes + self.binary_attributes + \
                self.nominal_attributes
        self.data_imputed= \
                np.hstack((self.imputed_interval_data,\
                           self.imputed_binary_data, \
                           self.imputed_nominal_data))
        self.imputed_data_df = \
                pd.DataFrame(self.data_imputed, columns=col)
    def scale_encode(self):
        self.standardize_interval()
        self.encode_binary()
        self.encode_nominal()
        self.encoded_data()
            
    def standardize_interval(self):
        if (self.n_interval==0 or self.interval_scale==None):
            self.scaled_interval_data = self.imputed_interval_data
            return
        # Standardize Interval Data using Z-Scores
        if self.interval_scale=='std':
            scaler = preprocessing.StandardScaler() 
            scaler.fit(self.imputed_interval_data)
            self.scaled_interval_data = \
                scaler.transform(self.imputed_interval_data)
        # Standardize Interval Data using median and IQR
        if self.interval_scale=='robust':
            scaler = preprocessing.RobustScaler() 
            scaler.fit(self.imputed_interval_data)
            self.scaled_interval_data = \
                scaler.transform(self.imputed_interval_data)
            
    def encode_binary(self):
        # Uses 1 and -1 encoding for binary instead of 0, 1
        # SAS uses the 1, -1 convention
        if self.n_binary == 0 or self.nominal_encoding == None:
            return
        if self.nominal_encoding == 'SAS':
            if self.n_binary == 0:
                return
            for j in range(self.n_binary):
                for i in range(self.n_obs):
                    if self.imputed_binary_data[i,j] == 1:
                        self.imputed_binary_data[i,j] = -1
                    else:
                        self.imputed_binary_data[i,j] = 1
              
    def encode_nominal(self):
        if (self.n_nominal==0 or self.nominal_encoding==None):
            return
        # Create an instance of the OneHotEncoder & Selecting Attributes
        # Attributes must all be non-negative integers
        # Missing values may show up as -1 values, which will cause an error
        onehot = preprocessing.OneHotEncoder()
        self.hot_array = \
                onehot.fit_transform(self.imputed_nominal_data).toarray()
        n_features = []
        nominal_categories = 0
        for i in range(self.n_nominal):
            feature = self.nominal_attributes[i]
            v = self.features_map[feature]
            n_features.append(len(v[1]))
            nominal_categories += len(v[1])
        if nominal_categories != self.hot_array.shape[1]:
            print("\n*****Error ->\nNumber of one-hot columns is", \
                  self.hot_array.shape[1], "but nominal ", \
                  "categories is ", nominal_categories)
            print("Data map probably contains nominal ", \
                  "categories not found in the data.\n")
            return None
            
        # SAS Encoding subtracts the last one-hot vector from the others, for
        # each nominal attribute.
        if self.nominal_encoding == 'SAS':
            self.sas_encoded = \
                np.zeros((self.n_obs, (self.n_onehot-self.n_nominal)))
            ilast = -1
            idx1  = 0
            idx2  = 0
            for l in range(self.n_nominal):
                m = n_features[l]
                ilast = ilast + m
                for j in range(m-1):
                        for i in range(self.n_obs):
                            last = self.hot_array[i,ilast]
                            self.sas_encoded[i,idx1] = \
                                            self.hot_array[i,idx2] - last
                        idx1 += 1
                        idx2 += 1
                idx2 += 1
            
            
    def encoded_data(self):
        # Bring encoded and scaled data together into a dataframe
        # The Imputed and Encoded Data
        if self.n_nominal==0:
            if self.interval_scale==None:
                self.data_encoded = np.hstack((self.imputed_interval_data, \
                                       self.imputed_binary_data))
            else:
                self.data_encoded = np.hstack((self.scaled_interval_data, \
                                       self.imputed_binary_data))
            if self.drop==True:
                for i in range(self.n_nominal):
                    self.col.remove(self.hot_drop_list[i])
                    
        if self.n_nominal>0 and self.nominal_encoding==None:
            if self.interval_scale==None:
                self.data_encoded = np.hstack((self.imputed_interval_data, \
                                       self.imputed_binary_data, \
                                       self.imputed_nominal_data))
            else:
                self.data_encoded = np.hstack((self.scaled_interval_data, \
                                       self.imputed_binary_data, \
                                       self.imputed_nominal_data))
                
        if self.n_nominal>0 and self.nominal_encoding == 'SAS':
            if self.interval_scale==None:
                self.data_encoded = np.hstack((self.imputed_interval_data, \
                                       self.imputed_binary_data, \
                                       self.sas_encoded))
            else:
                self.data_encoded = np.hstack((self.scaled_interval_data, \
                                       self.imputed_binary_data, \
                                       self.sas_encoded))
            if self.drop==True:
                for i in range(self.n_nominal):
                    self.col.remove(self.hot_drop_list[i])
                    
        if self.n_nominal>0 and self.nominal_encoding == 'one-hot':
            if self.interval_scale==None:
                self.data_encoded = np.hstack((self.imputed_interval_data, \
                                       self.imputed_binary_data, \
                                       self.hot_array))
            else:
                self.data_encoded = np.hstack((self.scaled_interval_data, \
                                       self.imputed_binary_data, \
                                       self.hot_array))
        # data_encoded array ready for conversion to dataframe
        self.encoded_data_df = \
                pd.DataFrame(self.data_encoded, columns=self.col)

        if self.nominal_encoding == 'one-hot' and self.drop==True:
            self.encoded_data_df = \
                self.encoded_data_df.drop(self.hot_drop_list, axis=1)
            for i in range(self.n_nominal):
                self.col.remove(self.hot_drop_list[i])
                
    def transform(self):
        self.impute()
        self.scale_encode()
        return self.encoded_data_df
        
    def fit_transform(self, df, data_map=None):
        self.fit(df, data_map)
        self.transform()
        return self.encoded_data_df
        