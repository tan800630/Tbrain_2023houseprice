import numpy as np
import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import LGBMRegressor

from utils import *

def preproc_data(dat, all_dist, ext, take_id_fns = []):
    df = dat.copy()
    
    # extra feature
    df['層數比例'] = df['移轉層次'] / df['總樓層數']

    # frequency encoding
    df['temp'] = df.apply(lambda row: '_'.join([row['縣市'], row['鄉鎮市區']]), axis = 1)
    ct = df['temp'].value_counts().to_frame().reset_index()
    ct.columns = ['temp', 'loc_count']
    df = df.merge(ct, on = 'temp', how = 'left')
    df.drop(columns = 'temp', inplace = True)
    
    ct = df['縣市'].value_counts().to_frame().reset_index()
    ct.columns = ['縣市', 'city_count']
    df = df.merge(ct, on = '縣市', how = 'left')
    
    # re-classify feature
    df['主要用途_cate'] = df['主要用途'].map(
        {'住家用':1, '集合住宅':1, '其他':2, '店鋪':2, '商業用':2, '國民住宅':1,'住工用':2,
         '一般事務所':2, '住商用':2, '廠房':2, '工業用':2, '辦公室':2})

    ## knn distance_mean (extra_feature)
    k=10
    kpar = np.partition(all_dist, kth = k)
    kth_dist_mean = kpar[:,:k].mean(axis = 1)
    df['10kth_dist_mean'] = kth_dist_mean
    
    
    # external data
    for s in ['國小', '國中', '高中']:
        ng = nearest_school(df, ext, s)
        df = pd.concat([df, ng], axis = 1)
    
    for s in ['醫療機構', '金融機構', '捷運站', '臺鐵站', 'ATM', '郵局', '公車站', '便利商店', 'MCD', '國道']:
        ng = nearest_general(df, ext, s, [300, 500, 1000])
        df = pd.concat([df, ng], axis = 1)
    
    for s in ['焚化爐', '機場', '高鐵站', '垃圾掩埋場', '汙水處理廠', '監獄', '工業區', '快速道路', '購物中心']:
        ng = nearest_general(df, ext, s)
        df = pd.concat([df, ng], axis = 1)
    
    for i, (threshold_ids, feature_fns) in enumerate(take_id_fns):
        all_nfx = nearest_feature(threshold_ids, df, feature_fns = feature_fns, hyphen = str(i))
        df = pd.concat([df, all_nfx], axis = 1)
    
    
    # extxl feature engineering
    df['area_ref_diff_路名_型態'] = df['area_路名_型態'] - df['建物面積']
    df['area_ref_diff_屋齡'] = df['area_屋齡'] - df['建物面積']
    df['area_ref_diff'] = df['area_精準'] - df['建物面積']
    df['area_ref_diff_路名_型態_屋齡'] = df['area_屋齡_型態_路名'] - df['建物面積']

    df['age_ref_diff_屋齡_型態_路名'] = df['age_屋齡_型態_路名'] - df['屋齡']
    df['age_ref_diff_路名_型態'] = df['age_路名_型態'] - df['屋齡']
    df['age_ref_diff'] = df['age_精準'] - df['屋齡']
    df['age_ref_diff_路名'] = df['age_路名'] - df['屋齡']

    df['price_diff_2021'] = df['price_路名_型態'] - df['price_路名_型態2020']
    df['price_diff_next'] = df['price_路名_型態'] - df['price_路名_型態2023']

    df['ref_diff'] = df['price_精準'] - df['price_路名_型態']
    df['ref_diff_精準_屋齡'] = df['price_精準'] - df['price_屋齡']
    df['ref_diff_cc'] = df['price_精準'] - df['price_屋齡_型態_路名']

    df['ref_price_屋齡_型態'] =  df['price_屋齡'] -  df['price_型態']
    df['ref_price_屋齡_屋齡_型態'] =  df['price_屋齡'] -  df['price_屋齡_型態']
    df['ref_price_屋齡_屋齡_型態_移轉'] =  df['price_屋齡'] -  df['price_屋齡_型態_移轉層次']
    df['ref_price_屋齡_屋齡_型態_移轉'] =  df['price_屋齡'] -  df['price_屋齡_型態_移轉層次']

    df['ref_price_路名_路名_型態'] =  df['price_路名'] - df['price_路名_型態']
    df['ref_price_路名_移轉'] =  df['price_路名'] - df['price_移轉']
    df['ref_price_路名_屋齡'] =  df['price_路名'] - df['price_屋齡']

    df['ref_price_路名_型態_型態'] =  df['price_路名_型態'] -  df['price_型態']
    df['ref_price_型態_移轉'] =  df['price_型態'] -  df['price_移轉']
    df['ref_price_路名_型態_路名_型態_屋齡'] =  df['price_路名_型態'] - df['price_屋齡_型態_路名']

    df['rev_路名_count_pro'] = df['len_路名_rev'] / (df['len_路名_rev'] + df['len_路名']+0.01)
    df['rev_型態_count_pro'] = df['len_型態_rev'] / (df['len_型態_rev'] + df['len_型態']+0.01)

    df['rev_路名_price_pro'] = df['price_路名_rev'] / (df['price_路名_rev'] + df['price_路名']+0.01)
    
    return df



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pred_filename", default = 'test_submission.csv')
    parser.add_argument("--mode", default = 'test', type=str)
    
    args = parser.parse_args()
    
    assert (args.mode=='test')|(args.mode=='validation')
    
    saved_name = args.test_pred_filename    
    test_mode = args.mode=='test'
    
    if test_mode:
        print('Testing mode : generate prediction file without printing validation score.')
    else:
        print('Validation mode : take 20% data to estimate mape score.')
    
    df = pd.read_csv('datasets/training_data.csv')
    test_df = pd.read_csv('datasets/public_dataset.csv')
    testp_df = pd.read_csv('datasets/private_dataset.csv')

    # outlier deletion
    df = df[~df['ID'].isin(['TR-5660' ,'TR-8800'])] 

    df.reset_index(drop = True, inplace = True)

    # load extra feature

    extxl = pd.read_csv('ext_data_processed/extxl.csv')

    del_cols = ['len_精準', 'floor_精準', 'parking_n_精準', 'height_精準', 'total_price_移轉', 'floor_移轉', 'parking_price_移轉',
    'parking_n_移轉', 'height_移轉', 'floor_路名_型態', 'height_路名_型態', 'floor_路名', 'parking_price_路名', 'parking_n_路名',
    'floor_型態', 'height_型態', 'len_屋齡', 'age_屋齡', 'total_price_屋齡', 'floor_屋齡', 'parking_price_屋齡', 'parking_n_屋齡', 'height_屋齡']

    extxl = extxl.drop(columns = del_cols)

    with open("ext_data_processed/ext.pkl", "rb") as f:
        ext = pickle.load(f)[0]

    with open("ext_data_processed/id_2_count_private.pkl", "rb") as f:
        id_2_count = pickle.load(f)


    if test_mode:
        df_all = pd.concat([df, test_df, testp_df], ignore_index=True)
    else:
        df_all = df

    df_all.reset_index(drop = True, inplace = True)


    # average distance of knn
    all_dist = distance_matrix(df_all[['縱坐標', '橫坐標']], df_all[['縱坐標', '橫坐標']])
    np.fill_diagonal(all_dist, all_dist.max())

    take_id_fns = []

    default_fn_dict = {'_mean':np.mean,'_max':np.max,'_min':np.min}

    f_func_default = [
        {
            'feature_value':n,
            'filter_features':[],
            'fn_dict':default_fn_dict} for n in ["屋齡", "總樓層數", "土地面積", "建物面積", "主建物面積", "陽台面積", "附屬建物面積"]
    ] + [
        {
            'feature_value':"屋齡",
            'feature_key':"屋齡",
            'filter_features':[],
            'fn_dict':{
                '_threshold_count':lambda values, key:len(values),
                '_difference':lambda values, key: key - np.mean(values),
                '_difference_min':lambda values, key: key - np.min(values),
                '_normalization':lambda values, key: (key-values.min())/(values.max() - values.min() + 0.1),
            }
        },{
            'feature_value':"總樓層數",
            'feature_key':"總樓層數",
            'filter_features':[],
            'fn_dict':{
                '_difference':lambda values, key: key - np.mean(values),
                '_difference_min':lambda values, key: key - np.min(values),
            }
        },{
            'feature_value':"屋齡",
            'feature_key':"屋齡",
            'filter_features':["建物型態"],
            'fn_dict':{
                '_difference_ftall':lambda values, key: key - np.mean(values),
                '_difference_min_ftall':lambda values, key: key - np.min(values),
                '_mean_ftall':lambda values, key: np.mean(values)
            }
        }, 
    ]

    # distance threshold 
    for dist_threshold in [500, 1000, 5000]:
        t_ids = [np.argwhere(d<dist_threshold).flatten() for d in all_dist]

        if dist_threshold==1000:
            take_id_fns.append((
                t_ids,f_func_default + [{
                    'feature_value':"路名",
                    'feature_key':"路名",
                    'filter_features':[],
                    'fn_dict':{
                        '_percent':lambda values, key:np.sum(values==key)/(len(values)+.01),
                    }}, {
                    'feature_value':"price_精準",
                    'filter_features':[],
                    'fn_dict':{
                        '_neighbor_mean':lambda values:np.sum(values)/(np.sum(values>0)+.01),
                        'neighbor_na_percent':lambda values:np.sum(values<0)/(len(values)+.01)
                }}, {
                    'feature_value':"price_屋齡",
                    'filter_features':[],
                    'fn_dict':{
                        '_neighbor_mean':lambda values:np.sum(values)/(np.sum(values>0)+.01),
                }}]
            ))
        else:
            take_id_fns.append((t_ids,f_func_default))

    df_allextxl = df_all.merge(extxl, on = 'ID', how = 'left')
    df_allextxl = df_allextxl.merge(id_2_count, on = 'ID')

    preproc_df = preproc_data(df_allextxl, all_dist, ext, take_id_fns=take_id_fns)


    preproc_id = preproc_df['ID']
    preproc_road = preproc_df['路名']

    preproc_df.drop(columns = ['ID', '備註' , '路名'], inplace = True)

    dummy_columns = None
    preproc_df = pd.get_dummies(data = preproc_df, columns = dummy_columns)

    preproc_df['路名'] = preproc_road


    if test_mode:

        train_x = preproc_df.iloc[:11749].drop(columns = '單價')
        test_x = preproc_df.iloc[11749:].drop(columns = '單價')

        train_y = preproc_df.iloc[:11749]['單價']
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(preproc_df.drop(columns = '單價'), 
                                                              preproc_df['單價'], test_size = 0.2, stratify = df['縣市'],
                                                              shuffle = True, random_state = 630) 
    if test_mode:
        valid_x = test_x.copy()

    dist_y = distance_matrix(pd.concat([train_x[["縱坐標", "橫坐標"]], valid_x[["縱坐標", "橫坐標"]]]),
                             pd.concat([train_x[["縱坐標", "橫坐標"]], valid_x[["縱坐標", "橫坐標"]]]))
    np.fill_diagonal(dist_y, dist_y.max())

    age_abs = np.abs(np.repeat(train_x['屋齡'].values, train_x.shape[0] + valid_x.shape[0]).reshape(
        train_x.shape[0], -1) - pd.concat([train_x['屋齡'], valid_x['屋齡']]).values).transpose()
    np.fill_diagonal(age_abs, age_abs.max())



    # mean_price - filtered by 建物型態, 總樓層數, 路名, 屋齡(threshold=2
    age_threshold = 2
    for threshold in [200, 1000]:

        train_threshold_ids = [np.argwhere(d<threshold).flatten() for d in dist_y[:train_x.shape[0], :train_x.shape[0]]]
        valid_threshold_ids = [np.argwhere(d<threshold).flatten() for d in dist_y[train_x.shape[0]:, :train_x.shape[0]]]

        train_age_ids = [np.argwhere(d<age_threshold).flatten() for d in age_abs[:train_x.shape[0], :train_x.shape[0]]]
        valid_age_ids = [np.argwhere(d<age_threshold).flatten() for d in age_abs[train_x.shape[0]:, :train_x.shape[0]]]

        train_join_ids = [t[np.isin(t,a)] for t,a in zip(train_threshold_ids, train_age_ids)]
        valid_join_ids = [t[np.isin(t,a)] for t,a in zip(valid_threshold_ids, valid_age_ids)]

        y_fns = [{
                        'feature_value':"單價",
                        'filter_features':["建物型態", "總樓層數", "路名"],
                        'fn_dict':{
                            'rs_mean':np.mean,
                        }}]

        train_xy = train_x.copy()
        train_xy['單價'] = train_y
        train_xy['建物型態'] = train_xy.filter(regex="建物型態").idxmax(axis = 1)

        valid_xy = valid_x.copy()
        valid_xy['建物型態'] = valid_xy.filter(regex="建物型態").idxmax(axis = 1)

        train_nfy = nearest_feature(train_threshold_ids, train_xy, y_fns, train_xy, hyphen = str(threshold))
        valid_nfy = nearest_feature(valid_threshold_ids, train_xy, y_fns, valid_xy, hyphen = str(threshold))

        train_agenfy = nearest_feature(train_join_ids, train_xy, y_fns, train_xy, hyphen = '_age'+str(threshold))
        valid_agenfy = nearest_feature(valid_join_ids, train_xy, y_fns, valid_xy, hyphen = '_age'+str(threshold))

        train_nfy = pd.concat([train_nfy, train_agenfy],1)
        valid_nfy = pd.concat([valid_nfy, valid_agenfy],1)

        for c in ['單價rs_mean' +str(threshold), '單價rs_mean_age'+str(threshold)]:

            t = train_nfy[c]
            v = valid_nfy[c]

            train_x[c] = t.values
            valid_x[c] = v.values


    # mean_price - filtered by 建物型態, 路名, 屋齡(threshold=5
    age_threshold = 5
    for threshold in [1000,]:

        train_threshold_ids = [np.argwhere(d<threshold).flatten() for d in dist_y[:train_x.shape[0], :train_x.shape[0]]]
        valid_threshold_ids = [np.argwhere(d<threshold).flatten() for d in dist_y[train_x.shape[0]:, :train_x.shape[0]]]

        train_age_ids = [np.argwhere(d<age_threshold).flatten() for d in age_abs[:train_x.shape[0], :train_x.shape[0]]]
        valid_age_ids = [np.argwhere(d<age_threshold).flatten() for d in age_abs[train_x.shape[0]:, :train_x.shape[0]]]

        train_join_ids = [t[np.isin(t,a)] for t,a in zip(train_threshold_ids, train_age_ids)]
        valid_join_ids = [t[np.isin(t,a)] for t,a in zip(valid_threshold_ids, valid_age_ids)]

        y_fns = [{
                        'feature_value':"單價",
                        'filter_features':["建物型態", "路名"],
                        'fn_dict':{
                            'r_mean':np.mean,
                        }}]
        train_xy = train_x.copy()
        train_xy['單價'] = train_y
        train_xy['建物型態'] = train_xy.filter(regex="建物型態").idxmax(axis = 1)

        valid_xy = valid_x.copy()
        valid_xy['建物型態'] = valid_xy.filter(regex="建物型態").idxmax(axis = 1)

        train_nfy = nearest_feature(train_threshold_ids, train_xy, y_fns, train_xy, hyphen = str(threshold))
        valid_nfy = nearest_feature(valid_threshold_ids, train_xy, y_fns, valid_xy, hyphen = str(threshold))

        train_agenfy = nearest_feature(train_join_ids, train_xy, y_fns, train_xy, hyphen = '_age'+str(threshold))
        valid_agenfy = nearest_feature(valid_join_ids, train_xy, y_fns, valid_xy, hyphen = '_age'+str(threshold))

        train_nfy = pd.concat([train_nfy, train_agenfy],1)
        valid_nfy = pd.concat([valid_nfy, valid_agenfy],1)

        for c in ['單價r_mean_age'+str(threshold), '單價r_mean_age'+str(threshold)]:

            t = train_nfy[c]
            v = valid_nfy[c]

            train_x[c] = t.values
            valid_x[c] = v.values

    if test_mode:
        test_x = valid_x.copy()

    if test_mode:
        valid_x = test_x.copy()

    for c in list(extxl.columns[1:]) + [
        '車位面積',
        'area_ref_diff_路名_型態', 'area_ref_diff', 'area_ref_diff_屋齡', 'area_ref_diff_路名_型態_屋齡',
        'age_ref_diff_路名_型態', 'age_ref_diff', 'age_ref_diff_屋齡_型態_路名', 'age_ref_diff_路名',
        'price_diff_2021', 'price_diff_next',
        'ref_diff', 'ref_diff_精準_屋齡', 'ref_diff_cc',
        'ref_price_屋齡_型態', 'ref_price_屋齡_屋齡_型態', 'ref_price_屋齡_屋齡_型態_移轉',
        'ref_price_路名_路名_型態', 'ref_price_路名_移轉', 'ref_price_路名_屋齡', 
        'ref_price_路名_型態_型態', 'ref_price_型態_移轉', 'ref_price_路名_型態_路名_型態_屋齡',
        'rev_路名_count_pro', 'rev_路名_price_pro',
        'rev_型態_count_pro', 
             ]: 
        quantile_ = quantize_feature(pd.concat([train_x[c], valid_x[c]]), q = 128)
        train_q = quantile_.iloc[:train_x.shape[0]]
        valid_q = quantile_.iloc[train_x.shape[0]:]

        train_x[c] = train_q
        valid_x[c] = valid_q

    if test_mode:
        test_x = valid_x.copy()

    # final data
    drop_columns = ['price_型態', "路名"]

    train_x = train_x.drop(columns = drop_columns)

    if test_mode:
        test_x = test_x.drop(columns = drop_columns)
    else:
        valid_x = valid_x.drop(columns = drop_columns)


    # model fitting
    m_ls = []
    test_pred_ls = []
    for i in range(6):

        if i%3==0:
            model = LGBMRegressor(n_estimators=10000, learning_rate=5e-2, reg_alpha = 1e-2, reg_lambda = 5e-1, max_depth=12,
                          importance_type="gain", objective='mse', verbose = -1, min_child_samples=3, random_state = 630+i,
                          subsample = 0.5, colsample_bytree=0.5, boosting_type = 'dart')
        elif i%3==1:
            model = LGBMRegressor(boosting_type='gbdt', n_estimators=10000, learning_rate=1e-2,num_leaves = 31, max_depth = -1,
                                  random_state = 630+i, min_child_samples=5, 
                              subsample=0.5, colsample_bytree=0.5, reg_alpha=3e-1, reg_lambda=3e-1, subsample_freq=4)
        else:
            model = LGBMRegressor(n_estimators=10000, learning_rate=1e-1, reg_alpha = 1e-1, reg_lambda = 5e-1, max_depth=12, random_state = 630+i,
                          importance_type="gain", objective='mse', drop_rate = 0.1, skip_drop = 0.8, max_drop = 50, xgboost_dart_mode = False,
                          subsample = 0.5, colsample_bytree=0.5, boosting_type = 'dart')


        model.fit(train_x, train_y)

        train_pred = model.predict(train_x)
        print('train error : ', mean_absolute_percentage_error(y_true = train_y, y_pred = train_pred))

        if test_mode:
            test_pred = model.predict(test_x)
            m_ls.append(model)
            test_pred_ls.append(test_pred)    
        else:
            valid_pred = model.predict(valid_x)
            print('valid error : ', mean_absolute_percentage_error(y_true = valid_y, y_pred = valid_pred))
            test_pred_ls.append(valid_pred)

    if not test_mode:
        print('valid error (ensemble) : ', mean_absolute_percentage_error(y_true = valid_y, y_pred = np.mean(test_pred_ls, axis = 0)))


    if test_mode:
        sample_submission = pd.read_csv('datasets/public_private_submission_template.csv')
        sample_submission['predicted_price'] = np.mean(test_pred_ls, axis = 0)
        sample_submission.to_csv(saved_name, index = False)
    

if __name__ == '__main__':

    main()