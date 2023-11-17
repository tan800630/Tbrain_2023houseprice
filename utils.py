import os
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import distance_matrix

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.utils import shuffle

def nearest_school(sample, ext, key, d_thresholds = [500]):
    
    assert key in ['國小', '國中', '高中', '大學']
    
    s_array = sample[['橫坐標', '縱坐標']].values.astype('int32')
    k_array = ext[key][['x', 'y']].values.astype('int32')
    dist = distance_matrix(s_array, k_array)
    
    school = ext[key].iloc[np.argmin(dist, axis = -1)]
    n_student = school.filter(regex='學生數').sum(axis = 1).values
    distance = dist.min(axis = -1)

    out_df = pd.DataFrame({key+'最近距離':dist.min(axis = -1), key+'學生數':n_student})

    for th in d_thresholds:
        out_df[str(th)+'公尺_'+key] = np.sum(dist<th, axis = 1)

    return out_df

def nearest_general(sample, ext, key, d_thresholds = []):
    
    s_array = sample[['橫坐標', '縱坐標']].values.astype('int32')
    k_array = ext[key][['x', 'y']].dropna().values.astype('int32')
    
    dist = distance_matrix(s_array, k_array)
    out_df = pd.DataFrame({key+'最近距離':dist.min(axis = -1)})
        
    th_out = []
    for th in d_thresholds:
        th_out.append(np.sum(dist<th, axis = 1))
        out_df[str(th)+'公尺_'+key] = np.sum(dist<th, axis = 1)
    
    return out_df

def split_by_age(dat):
    df = dat.copy()
    df = df.sort_values('屋齡')
    
    age_values = df['屋齡'].values
    id_values = df['ID'].values
    
    split_ls = [0]
    max_diff = 0
    max_i = 0
    for i in range(1,df.shape[0]):
        diff = age_values[i] - age_values[i-1]
        if diff>2:
            split_ls.append(i) # i前砍
        elif (age_values[i] - age_values[split_ls[-1]])>2:
            split_ls.append(i)
    
    split_ls.append(df.shape[0])
    
    out_ls = []
    for i in range(1, len(split_ls)):
        out_ls.append(id_values[split_ls[i-1]:split_ls[i]])
    
    return out_ls

def nearest_feature(take_id, dat, feature_fns, datb = None, hyphen = ''):
    
    '''
    feature_fns : list of dicts. keys in dict include : feature_value, (feature_key), filter_features, fn_dict(also a dict. keys will be used as columnname)
    同個特徵若有需要以不同方式篩選資料需要再列一次
    '''
    
    df = dat.reset_index()
    if datb is None:
        dfb = dat.reset_index()
    else:
        dfb = datb.reset_index()
    out_dict = {}
    
    for i, n in enumerate(take_id):
        sub_df = df.loc[n,:]
        
        for feature_setting in feature_fns:
            fns = feature_setting['fn_dict']
            take_df = sub_df.copy()
            
            if 'feature_key' in feature_setting.keys():
                k_value = dfb[feature_setting['feature_key']].iloc[i]
            f_v = feature_setting['feature_value']
            
            # filting data
            for f in feature_setting['filter_features']:
                take_df = take_df[take_df[f]==dfb[f].iloc[i]]
            
            for func_name, func in fns.items():
                if f_v+func_name not in out_dict.keys():
                    out_dict[f_v+func_name] = []
                    
                if len(sub_df)>0:
                    if 'feature_key' in feature_setting.keys():
                        out_dict[f_v+func_name].append(func(take_df[f_v], k_value))
                    else:
                        out_dict[f_v+func_name].append(func(take_df[f_v]))
                else:
                    out_dict[f_v+func_name].append(-1)
    out_df = pd.DataFrame(out_dict).fillna(-1)
    out_df.columns = out_df.columns + hyphen
    return out_df

def quantize_feature(df_x, q):
    quantile_x = pd.qcut(df_x, q, labels = False, duplicates='drop').astype('int')
    return quantile_x

def lonlat_to_97(lon,lat):
    import numpy as np
    """
    It transforms longitude, latitude to TWD97 system.
    
    https://tylerastro.medium.com/twd97-to-longitude-latitude-dde820d83405

    Parameters
    ----------
    lon : float
        longitude in degrees
    lat : float
        latitude in degrees 

    Returns
    -------
    x, y [TWD97]
    """
    
    lat = np.radians(lat)
    lon = np.radians(lon)
    
    a = 6378137.0
    b = 6356752.314245
    long0 = np.radians(121)
    k0 = 0.9999
    dx = 250000

    e = (1 - b ** 2 / a ** 2) ** 0.5
    e2 = e ** 2 / (1 - e ** 2)
    n = (a - b) / (a + b)
    nu = a / (1 - (e ** 2) * (np.sin(lat) ** 2)) ** 0.5
    p = lon - long0

    A = a * (1 - n + (5 / 4.0) * (n ** 2 - n ** 3) + (81 / 64.0) * (n ** 4  - n ** 5))
    B = (3 * a * n / 2.0) * (1 - n + (7 / 8.0) * (n ** 2 - n ** 3) + (55 / 64.0) * (n ** 4 - n ** 5))
    C = (15 * a * (n ** 2) / 16.0) * (1 - n + (3 / 4.0) * (n ** 2 - n ** 3))
    D = (35 * a * (n ** 3) / 48.0) * (1 - n + (11 / 16.0) * (n ** 2 - n ** 3))
    E = (315 * a * (n ** 4) / 51.0) * (1 - n)

    S = A * lat - B * np.sin(2 * lat) + C * np.sin(4 * lat) - D * np.sin(6 * lat) + E * np.sin(8 * lat)

    K1 = S * k0
    K2 = k0 * nu * np.sin(2 * lat) / 4.0
    K3 = (k0 * nu * np.sin(lat) * (np.cos(lat) ** 3) / 24.0) * (5 - np.tan(lat) ** 2 + 9 * e2 * (np.cos(lat) ** 2) + 4 * (e2 ** 2) * (np.cos(lat) ** 4))

    y_97 = K1 + K2 * (p ** 2) + K3 * (p ** 4)

    K4 = k0 * nu * np.cos(lat)
    K5 = (k0 * nu * (np.cos(lat) ** 3) / 6.0) * (1 - np.tan(lat) ** 2 + e2 * (np.cos(lat) ** 2))

    x_97 = K4 * p + K5 * (p ** 3) + dx
    
    return x_97, y_97

def twd97_to_lonlat(x=174458.0,y=2525824.0):
    import math
    """
    Parameters
    ----------
    x : float
        TWD97 coord system. The default is 174458.0.
    y : float
        TWD97 coord system. The default is 2525824.0.
    Returns
    -------
    list
        [longitude, latitude]
    """
    
    a = 6378137
    b = 6356752.314245
    long_0 = 121 * math.pi / 180.0
    k0 = 0.9999
    dx = 250000
    dy = 0
    
    e = math.pow((1-math.pow(b, 2)/math.pow(a,2)), 0.5)
    
    x -= dx
    y -= dy
    
    M = y / k0
    
    mu = M / ( a*(1-math.pow(e, 2)/4 - 3*math.pow(e,4)/64 - 5 * math.pow(e, 6)/256))
    e1 = (1.0 - pow((1   - pow(e, 2)), 0.5)) / (1.0 +math.pow((1.0 -math.pow(e,2)), 0.5))
    
    j1 = 3*e1/2-27*math.pow(e1,3)/32
    j2 = 21 * math.pow(e1,2)/16 - 55 * math.pow(e1, 4)/32
    j3 = 151 * math.pow(e1, 3)/96
    j4 = 1097 * math.pow(e1, 4)/512
    
    fp = mu + j1 * math.sin(2*mu) + j2 * math.sin(4* mu) + j3 * math.sin(6*mu) + j4 * math.sin(8* mu)
    
    e2 = math.pow((e*a/b),2)
    c1 = math.pow(e2*math.cos(fp),2)
    t1 = math.pow(math.tan(fp),2)
    r1 = a * (1-math.pow(e,2)) / math.pow( (1-math.pow(e,2)* math.pow(math.sin(fp),2)), (3/2))
    n1 = a / math.pow((1-math.pow(e,2)*math.pow(math.sin(fp),2)),0.5)
    d = x / (n1*k0)
    
    q1 = n1* math.tan(fp) / r1
    q2 = math.pow(d,2)/2
    q3 = ( 5 + 3 * t1 + 10 * c1 - 4 * math.pow(c1,2) - 9 * e2 ) * math.pow(d,4)/24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * math.pow(t1,2) - 3 * math.pow(c1,2) - 252 * e2) * math.pow(d,6)/720
    lat = fp - q1 * (q2 - q3 + q4)
    
    
    q5 = d
    q6 = (1+2*t1+c1) * math.pow(d,3) / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * math.pow(c1,2) + 8 * e2 + 24 * math.pow(t1,2)) * math.pow(d,5) / 120
    lon = long_0 + (q5 - q6 + q7) / math.cos(fp)
    
    lat = (lat*180) / math.pi
    lon = (lon*180) / math.pi
    return lon, lat
