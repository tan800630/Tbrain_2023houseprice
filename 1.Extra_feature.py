import os
import re
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import skew

from utils import *

def parse_date(s):
    
    if s=='0800230':
        return '1991/12/30'
    sp=2
    if (s[0]=='1')|(s[0]=='0'):
        sp = 3
    year = s[:sp]
    month = s[sp:sp+2]
    day = s[sp+2:sp+4]
    
    if year=='na':
        return '1911/01/01'
    elif month=='00':
        month='01'
    elif (day=='00')|(day==''):
        day = '01'
    else:
        return str(1911+int(year))+'/'+month+'/'+day



def main():
    df = pd.read_csv('datasets/training_data.csv')

    # outlier deletion
    df = df[~df['ID'].isin(['TR-5660' ,'TR-8800'])]  #

    df.reset_index(drop = True, inplace = True)

    test_df = pd.read_csv('datasets/public_dataset.csv')
    testp_df = pd.read_csv('datasets/private_dataset.csv')

    df_all = pd.concat([df, test_df, testp_df], axis = 0, ignore_index=True)
    df_all.reset_index(drop = True, inplace = True)

    # 周邊設施
    dir_ = 'external_data/周邊設施/'

    ext_list = []
    for name in ['高中基本資料', '國中基本資料', '國小基本資料', '捷運站點資料', '火車站點資料', '高鐵站點資料', '公車站點資料',
                 '便利商店', '金融機構基本資料', '郵局據點資料', 'ATM資料', '醫療機構基本資料', '麥當勞門市資料',
                 '快速公路交流道里程及通往地名_11209', '監獄資料_經緯度', '垃圾掩埋場基本資料', '汙水處理廠資料_經緯度', '焚化爐基本資料_經緯度',
                 '機場資料', '工業區資料', '購物中心資料_經緯度', '國道交通流道_估計']:

        df = pd.read_csv(os.path.join(dir_, name+'.csv'), header=0)

        if name=='垃圾掩埋場基本資料':
            df.rename({'latitude':'lat', 'longitude':'lng'}, axis = 1, inplace = True)

        ext_list.append(df)

    dict_key = ['高中','國中', '國小', '捷運站', '臺鐵站', '高鐵站', '公車站', '便利商店', '金融機構', '郵局', 'ATM', '醫療機構',
                'MCD', '快速道路', '監獄', '垃圾掩埋場', '汙水處理廠', '焚化爐', '機場', '工業區', '購物中心', '國道']


    ext = {k:v for k,v in zip(dict_key, ext_list)}

    for key, values in ext.items():
        if key=='國道':
            x, y = values['坐標X-TWD97'], values['坐標Y-TWD97']
        elif key=='快速道路':
            x, y = values['TWD97-X'], values['TWD97-Y']
        else:
            x, y = lonlat_to_97(values['lng'], values['lat'])
        ext[key]['x'] = x
        ext[key]['y'] = y


    # save processed file
    with open("ext_data_processed/ext.pkl", "wb") as f:
        pickle.dump([ext], f)


    # id_2_count
    out_list = []
    dist_threshold = 150
    age_threshold = 2
    for i, group_df in df_all.groupby(['縣市', '鄉鎮市區', '路名', '總樓層數', '主要建材', '主要用途']):
        group_df = group_df.sort_values('屋齡')
        distance = distance_matrix(group_df[['縱坐標', '橫坐標']], group_df[['縱坐標', '橫坐標']])

        del_idx = []
        age_values = group_df['屋齡'].values
        id_values = group_df['ID'].values
        while True:

            if len(id_values)==1:
                out_list.append(id_values)
                break

            t_age = age_values[0]

            mask = (age_values <= (age_values[0]+age_threshold)) & ((distance[0,:] - distance[0, 0])<=dist_threshold)
            out_list.append(id_values[mask])

            distance = distance[~mask,:][:, ~mask]
            age_values = age_values[~mask]
            id_values = id_values[~mask]

            if len(id_values)==0:
                break


    id_list = []
    count_list = []
    age_range_list, age_per_list, age_diff_list, age_sd_list = [], [], [] , []
    for id_ls in out_list:
        ct = len(id_ls) # 同棟大樓有幾筆資料

        sub_df = df_all[df_all['ID'].isin(id_ls)].sort_values('屋齡')

        min_age, max_age = sub_df['屋齡'].min(), sub_df['屋齡'].max()
        for i in range(sub_df.shape[0]):
            id_list.append(sub_df['ID'].values[i])
            count_list.append(ct)
            age_per_list.append((sub_df['屋齡'].values[i]-min_age) / (max_age - min_age + 0.01))
            age_range_list.append(max_age - min_age)
            age_diff_list.append(sub_df['屋齡'].mean() - sub_df['屋齡'].iloc[i])
            # age_sd_list.append(sd)

    id_2_count = pd.DataFrame({
        'ID':id_list,
        'same_building_count':count_list,
        'age_percent':age_per_list,
        'age_range':age_range_list,
        'age_diff':age_diff_list,
    })

    with open("ext_data_processed/id_2_count_private.pkl", "wb") as f:
        pickle.dump(id_2_count, f)


    # 實價登錄 2021-2022
    dir_ = 'ext_data_processed/'

    extxl = pd.read_csv(os.path.join(dir_, '實價登錄資料_處理後_2021-2022.csv'))

    extxl = extxl[extxl["買賣類別"]=='不動產買賣']
    extxl = extxl[extxl["建物型態"].isin(['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)', '公寓(5樓含以下無電梯)'])]
    extxl = extxl[extxl["非都市土地使用分區"].isna()]

    extxl['移轉層次'] = extxl['移轉層次_數字'].astype('int')
    extxl['總樓層數'] = extxl['總樓層數_數字'].astype('int')
    extxl['移轉層次_cate'] = (extxl['移轉層次']>5)*1 + (extxl['移轉層次']>10)*1
    extxl['交易年月日'] = pd.to_datetime(extxl['交易年月日'])
    extxl = extxl[extxl['交易年月日'].dt.year.isin([2021, 2022])]
    extxl = extxl[extxl["移轉層次"]>1]

    for c in ["建物移轉總面積平方公尺"]:
        extxl[c] = (extxl[c] - extxl[c].mean())/extxl[c].std()
        extxl = extxl[extxl[c]<10]
        extxl[c] = (extxl[c] - extxl[c].mean())/extxl[c].std()

    extxl = extxl[~extxl['備註'].fillna('').str.contains('特殊關係')]

    category_ = {
        '精準':["縣市", '建物型態', '移轉層次', '總樓層數', "路名", "屋齡.5"],
        '移轉':["縣市", "建物型態", "移轉層次_cate"],
        '屋齡_型態':["縣市", "屋齡3", "建物型態"],
        '屋齡_型態_移轉層次':["縣市", "屋齡3", "建物型態", "移轉層次_cate"], #
        '屋齡_型態_路名':["縣市", '屋齡3', '建物型態', '路名'],
        '同建物':["縣市", "建物型態", "總樓層數", "路名", "屋齡.5"],
        '路名_型態':["縣市", '建物型態', '路名'],
        '路名':["縣市", '路名'],
        '型態':["縣市", '建物型態'],
        '屋齡':["縣市", '屋齡3'],
    }
    cate_out = {}

    for k in category_.keys():
        cate_out[k] = {
            'len':np.ones(df_all['ID'].shape[0]),
            'price':np.ones(df_all['ID'].shape[0]),
            'price_sd':np.ones(df_all['ID'].shape[0]),
            'price_skew':np.ones(df_all['ID'].shape[0]),
            'age':np.ones(df_all['ID'].shape[0]),
            'area':np.ones(df_all['ID'].shape[0]),
            'total_price':np.ones(df_all['ID'].shape[0]),
            'floor':np.ones(df_all['ID'].shape[0]),
            'parking_price':np.ones(df_all['ID'].shape[0]),
            'parking_n':np.ones(df_all['ID'].shape[0]),
            'area_percent':np.ones(df_all['ID'].shape[0]),
            'height':np.ones(df_all['ID'].shape[0]),
            'room':np.ones(df_all['ID'].shape[0]),
            'ID':np.ones(df_all['ID'].shape[0], dtype = 'object'),
                      }

    mask_list = ["縣市", '建物型態', '移轉層次', '總樓層數', "路名", "屋齡3", "移轉層次_cate", "屋齡.5"]
    j=0
    for district, group_df in df_all.groupby('鄉鎮市區'):

        g_extxl = extxl[extxl["鄉鎮市區"]==district]    
        group_df['移轉層次_cate'] = (group_df['移轉層次']>5)*1 + (group_df['移轉層次']>10)*1

        for i in range(group_df.shape[0]):

            mask_dict = {}

            for c in mask_list:
                if c=='路名':
                    mask_dict[c] = (g_extxl["土地位置建物門牌"].str.contains(group_df["路名"].iloc[i]))*1
                elif c[:2]=='屋齡':
                    t=float(c[2:])
                    mask_dict[c] = ((g_extxl["屋齡"]>=(group_df["屋齡"].iloc[i]-t))&(g_extxl["屋齡"]<=(group_df["屋齡"].iloc[i]+t))).astype('int')
                else:
                    mask_dict[c] = (g_extxl[c]==group_df[c].iloc[i])*1

            for name, map_cols in category_.items():

                mask = np.zeros(g_extxl.shape[0])

                for col in map_cols:
                    mask += mask_dict[col]

                gmdf = g_extxl[mask==len(map_cols)]

                cate_out[name]['len'][j] = gmdf.shape[0]
                cate_out[name]['price'][j] = gmdf['單價元平方公尺'].mean()
                cate_out[name]['price_sd'][j] = gmdf['單價元平方公尺'].std()
                cate_out[name]['price_skew'][j] = skew(gmdf['單價元平方公尺'])
                cate_out[name]['age'][j] = gmdf['屋齡'].mean()
                cate_out[name]['total_price'][j] = gmdf['總價元'].mean()
                cate_out[name]['height'][j] = gmdf["總樓層數"].mean()
                cate_out[name]['area'][j] = gmdf['建物移轉總面積平方公尺'].mean()
                cate_out[name]['floor'][j] = gmdf['移轉層次'].mean()
                cate_out[name]['parking_price'][j] = gmdf['車位總價元'].mean()
                cate_out[name]['parking_n'][j] = gmdf['車位個數'].mean()
                cate_out[name]['area_percent'][j] = (gmdf['建物移轉總面積平方公尺'] > group_df['建物面積'].iloc[i]).mean()
                cate_out[name]['room'][j] = gmdf['建物現況格局-房'].mean()

                cate_out[name]['ID'][j] = group_df['ID'].iloc[i]
            j+=1

    df_list = [(name, pd.DataFrame(dict_).fillna(-1)) for name, dict_ in cate_out.items()]

    for (name, df) in df_list:
        df.columns = [c+'_'+name if c!='ID' else c for c in df.columns]

    total_df = pd.DataFrame({'ID':df_list[0][1]['ID']})

    for (name, df) in df_list:
        total_df = total_df.merge(df, on = 'ID')


    # 實價登錄 2021-2022 非不動產
    dir_ = 'ext_data_processed/'

    extxl = pd.read_csv(os.path.join(dir_, '實價登錄資料_處理後_2021-2022.csv'))
    rev_extxl = extxl[extxl['買賣類別']!='不動產買賣']
    rev_extxl = rev_extxl[~rev_extxl['都市土地使用分區'].isna()]

    rev_extxl = rev_extxl[rev_extxl['移轉層次_數字'].isin([str(i) for i in range(50)])]
    rev_extxl['移轉層次'] = rev_extxl['移轉層次_數字'].astype('int')
    rev_extxl['交易年月日'] = pd.to_datetime(rev_extxl['交易年月日'])
    rev_extxl = rev_extxl[rev_extxl['交易年月日'].dt.year.isin([2021, 2022])]
    extxl = rev_extxl.copy()

    category_ = {
        '路名':["縣市", '路名'],
        '型態':["縣市", '建物型態'],
    }
    cate_out = {}

    for k in category_.keys():
        cate_out[k] = {
            'len':np.ones(df_all['ID'].shape[0]),
            'price':np.ones(df_all['ID'].shape[0]),
            'area':np.ones(df_all['ID'].shape[0]),
            'total_price':np.ones(df_all['ID'].shape[0]),
            'parking_price':np.ones(df_all['ID'].shape[0]),
            'parking_n':np.ones(df_all['ID'].shape[0]),
            'room':np.ones(df_all['ID'].shape[0]),
            'ID':np.ones(df_all['ID'].shape[0], dtype = 'object'),
                      }
    mask_list = ["縣市", '建物型態', "路名"]
    j=0
    for district, group_df in df_all.groupby('鄉鎮市區'):

        g_extxl = extxl[extxl["鄉鎮市區"]==district]    
        group_df['移轉層次_cate'] = (group_df['移轉層次']>5)*1 + (group_df['移轉層次']>10)*1

        for i in range(group_df.shape[0]):

            mask_dict = {}

            for c in mask_list:
                if c=='路名':
                    mask_dict[c] = (g_extxl["土地位置建物門牌"].str.contains(group_df["路名"].iloc[i]))*1
                elif c[:2]=='屋齡':
                    t=float(c[2:])
                    mask_dict[c] = ((g_extxl["屋齡"]>=(group_df["屋齡"].iloc[i]-t))&(g_extxl["屋齡"]<=(group_df["屋齡"].iloc[i]+t))).astype('int')
                else:
                    mask_dict[c] = (g_extxl[c]==group_df[c].iloc[i])*1

            for name, map_cols in category_.items():

                mask = np.zeros(g_extxl.shape[0])

                for col in map_cols:
                    mask += mask_dict[col]

                gmdf = g_extxl[mask==len(map_cols)]

                cate_out[name]['len'][j] = gmdf.shape[0]
                cate_out[name]['price'][j] = gmdf['單價元平方公尺'].mean()
                cate_out[name]['total_price'][j] = gmdf['總價元'].mean()
                cate_out[name]['area'][j] = gmdf['建物移轉總面積平方公尺'].mean()
                cate_out[name]['parking_price'][j] = gmdf['車位總價元'].mean()
                cate_out[name]['parking_n'][j] = gmdf['車位個數'].mean()
                cate_out[name]['room'][j] = gmdf['建物現況格局-房'].mean()

                cate_out[name]['ID'][j] = group_df['ID'].iloc[i]
            j+=1

    df_list = [(name, pd.DataFrame(dict_).fillna(-1)) for name, dict_ in cate_out.items()]

    for (name, df) in df_list:
        df.columns = [c+'_'+name+'_rev' if c!='ID' else c for c in df.columns]

    totalrev_df = pd.DataFrame({'ID':df_list[0][1]['ID']})

    for (name, df) in df_list:
        totalrev_df = totalrev_df.merge(df, on = 'ID')


    # 實價登錄 - 2020
    dir_ = 'ext_data_processed/'

    extxl = pd.read_csv(os.path.join(dir_, '實價登錄資料_2020全年.csv'))

    extxl = extxl[extxl["建物型態"].isin(['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)', '公寓(5樓含以下無電梯)'])]
    extxl = extxl[extxl['交易標的'].isin(['房地(土地+建物)', '房地(土地+建物)+車位', '建物'])]
    extxl = extxl[extxl["非都市土地使用分區"].isna()]

    extxl['移轉層次_rep'] = extxl.移轉層次.str.replace('平台，|，車庫|停車場，|陽台，|騎樓，|走廊，|通道，|，瞭望台|陽臺，|電梯樓梯間，|，停車場|，見其它登記事項|，儲藏室|，瞭望室|，通道|，門廳|，走廊|，防空避難室|，平台|，騎樓|，陽臺|，見使用執照|，見其他登記事項|，屋頂突出物|，夾層|，電梯樓梯間|，陽台|，機械房|，通道|見其他登記事項，|，露台|露台，|夾層，|屋頂突出物|見使用執照|見其他登記事項|防空避難室|夾層|見其它登記事項|管理員室（警衛室）',
        '', regex = True)
    extxl = extxl[~extxl['移轉層次_rep'].str.contains(',|，', regex = True).fillna(False)]
    extxl = extxl[~extxl.移轉層次_rep.str.contains('[^十下]一', regex = True).fillna(False)]
    extxl = extxl[~extxl.移轉層次_rep.str.contains('地下', regex = True).fillna(False)]

    extxl = extxl[extxl['交易年月日'].fillna(-1).apply(lambda s: int(str(s)[:3]))==109]
    extxl['交易年月日'] = pd.to_datetime(extxl['交易年月日'].astype('str').apply(
        lambda s: str(int(s[:3])+1911)+'/'+s[3:5]+'/'+s[5:] if (s[3:5]!='00') else str(int(s[:3])+1911)+'/1/1'))
    extxl['建築完成年月'] = pd.to_datetime(extxl['建築完成年月'].astype('str').fillna('1911/01/01').apply(parse_date))
    extxl['屋齡'] = round((extxl['交易年月日'] - extxl['建築完成年月']).dt.days/365,1)

    extxl['總價元'] = extxl['總價元'].astype('float')
    extxl['單價元平方公尺'] = extxl['單價元平方公尺'].astype('float')
    extxl['建物移轉總面積平方公尺'] = extxl['建物移轉總面積平方公尺'].astype('float')

    for c in ["建物移轉總面積平方公尺"]:
        extxl[c] = (extxl[c] - extxl[c].mean())/extxl[c].std()
        extxl = extxl[extxl[c]<10]
        extxl[c] = (extxl[c] - extxl[c].mean())/extxl[c].std()

    extxl = extxl[~extxl['備註'].fillna('').str.contains('特殊關係')]

    # 笨笨的手動處理總樓層數
    extxl['總樓層數'] = extxl.總樓層數.map({'二層':2, '十一層':11, '九層':9, '十四層':14, '七層':7, '十二層':12, '八層':8, '二十三層':23,'十層':10, '二十層':20,
           '十五層':15, '四層':4, '五層':5, '十八層':18, '六層':6, '二十一層':21, '十三層':13, '十六層':16, '二十七層':27, '三層':3,
           '十九層':19, '二十四層':24, '十七層':17, '二十六層':26, '二十五層':25, '二十二層':22, '二十九層':29, '四十二層':42,
           '二十八層':28, '三十層':30, '三十三層':33, '三十八層':38, '三十一層':31, '12':12, '13':13, '7':7, '6':6, '21':21,
           '15':15, '9':9, '22':22, '11':11, '三十二層':32, '三十九層':39, '三十五層':35, '四十一層':41, '三十四層':34,
           '一層':1, '三十七層':37, '17':17, '14':14, '24':24, '八十五層':85, '五十層':50, '三十六層':36, '六十八層':68,
           '四十層':40, '四十三層':43, '四十六層':46, '29':29, '26':26, '000':0, '見其他登記事項':0, '8':8, '5':5,
           '18':18, '19':19, '68':68, '10':10, '20':20, '見使用執照':0, '25':25, '30':30, '27':27, '16':16,
           '31':31, '23':23, '28':28})

    extxl = extxl[extxl["總樓層數"]!=1]

    category_ = {
        '路名_型態':["縣市", '建物型態', '路名'],
        '同建物':["縣市", "建物型態", "總樓層數", "路名"],
        '精準':["縣市", "建物型態", "總樓層數", "路名", '屋齡2020'],

    }
    cate_out = {}

    for k in category_.keys():
        cate_out[k] = {
            'len':np.ones(df_all['ID'].shape[0]),
            'price':np.ones(df_all['ID'].shape[0]),
            'area':np.ones(df_all['ID'].shape[0]),
            'total_price':np.ones(df_all['ID'].shape[0]),
            'ID':np.ones(df_all['ID'].shape[0], dtype = 'object'),
                      }
    mask_list = ["縣市", '建物型態', "路名","總樓層數", "屋齡2020"]
    j=0
    for district, group_df in df_all.groupby('鄉鎮市區'):

        g_extxl = extxl[extxl["鄉鎮市區"]==district]    
        group_df['移轉層次_cate'] = (group_df['移轉層次']>5)*1 + (group_df['移轉層次']>10)*1

        for i in range(group_df.shape[0]):

            mask_dict = {}

            for c in mask_list:
                if c=='路名':
                    mask_dict[c] = (g_extxl["土地位置建物門牌"].str.contains(group_df["路名"].iloc[i]))*1
                elif c=='屋齡2020':
                    mask_dict[c] = ((g_extxl["屋齡"]>=(group_df["屋齡"].iloc[i]-3))&(g_extxl["屋齡"]<=(group_df["屋齡"].iloc[i]))).astype('int')
                else:
                    mask_dict[c] = (g_extxl[c]==group_df[c].iloc[i])*1

            for name, map_cols in category_.items():

                mask = np.zeros(g_extxl.shape[0])

                for col in map_cols:
                    mask += mask_dict[col]

                gmdf = g_extxl[mask==len(map_cols)]

                cate_out[name]['len'][j] = gmdf.shape[0]
                cate_out[name]['price'][j] = gmdf['單價元平方公尺'].mean()
                cate_out[name]['total_price'][j] = gmdf['總價元'].mean()
                cate_out[name]['area'][j] = gmdf['建物移轉總面積平方公尺'].mean()

                cate_out[name]['ID'][j] = group_df['ID'].iloc[i]
            j+=1

    df_list = [(name, pd.DataFrame(dict_).fillna(-1)) for name, dict_ in cate_out.items()]

    for (name, df) in df_list:
        df.columns = [c+'_'+name+'2020' if c!='ID' else c for c in df.columns]

    total2020_df = pd.DataFrame({'ID':df_list[0][1]['ID']})

    for (name, df) in df_list:
        total2020_df = total2020_df.merge(df, on = 'ID')


    # 實價登錄 - 2023 q1-q3
    dir_ = 'ext_data_processed/'

    extxl = pd.read_csv(os.path.join(dir_, '實價登錄資料_2023q1q3.csv'))

    extxl = extxl[extxl["建物型態"].isin(['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)', '公寓(5樓含以下無電梯)'])]
    extxl = extxl[extxl['交易標的'].isin(['房地(土地+建物)', '房地(土地+建物)+車位', '建物'])]
    extxl = extxl[extxl["非都市土地使用分區"].isna()]

    extxl['移轉層次_rep'] = extxl.移轉層次.str.replace(
        '，倉庫|，浴廁|平台，|，車庫|停車場，|陽台，|騎樓，|走廊，|通道，|，瞭望台|陽臺，|電梯樓梯間，|，停車場|，見其它登記事項|，儲藏室|，瞭望室|，通道|，門廳|，走廊|，防空避難室|，平台|，騎樓|，陽臺|，見使用執照|，見其他登記事項|，屋頂突出物|，夾層|，電梯樓梯間|，陽台|，機械房|，通道|見其他登記事項，|，露台|露台，|夾層，|屋頂突出物|見使用執照|見其他登記事項|防空避難室|夾層|見其它登記事項|管理員室（警衛室）|倉庫|1樓+2樓|1樓至4樓|1-4樓|1-3樓|地上001層、001層',
        '', regex = True)
    extxl = extxl[~extxl['移轉層次_rep'].str.contains(',|，|加', regex = True).fillna(False)]
    extxl = extxl[~extxl.移轉層次_rep.str.contains('[^十]一', regex = True).fillna(False)]
    extxl = extxl[~extxl.移轉層次_rep.str.contains('^一|001|地下|全', regex = True).fillna(False)]

    extxl['總價元'] = extxl['總價元'].astype('float')
    extxl['單價元平方公尺'] = extxl['單價元平方公尺'].astype('float')
    extxl['建物移轉總面積平方公尺'] = extxl['建物移轉總面積平方公尺'].astype('float')

    extxl = extxl[extxl['交易年月日'].fillna(-1).apply(lambda s: int(str(s)[:3]))==112]
    extxl['交易年月日'] = pd.to_datetime(extxl['交易年月日'].astype('str').apply(
        lambda s: str(int(s[:3])+1911)+'/'+s[3:5]+'/'+s[5:] if (s[3:5]!='00') else str(int(s[:3])+1911)+'/1/1'))
    extxl['建築完成年月'] = pd.to_datetime(extxl['建築完成年月'].astype('str').fillna('1911/01/01').apply(parse_date))
    extxl['屋齡'] = round((extxl['交易年月日'] - extxl['建築完成年月']).dt.days/365,1)

    extxl = extxl[~extxl['備註'].fillna('').str.contains('特殊關係')]

    for c in ["建物移轉總面積平方公尺"]:
        extxl[c] = (extxl[c] - extxl[c].mean())/extxl[c].std()
        extxl = extxl[extxl[c]<10]
        extxl[c] = (extxl[c] - extxl[c].mean())/extxl[c].std()


    extxl['總樓層數'] = extxl.總樓層數.map({'九層':9, '十層':10, '五層':5, '七層':7, '二層':2, '八層':8, '四層':4, '十三層':13, '十二層':12, '十一層':11,
           '六層':6, '十四層':14, '十七層':17, '二十四層':24, '十五層':15, '十六層':16, '十八層':18, '二十一層':21, '三層':3,
           '十九層':19, '二十五層':25, '二十二層':22, '二十八層':28, '二十三層':23, '二十六層':26, '二十七層':27, '二十層':20,
           '三十一層':31, '二十九層':29, '三十層':30, '三十三層':33, '三十二層':32, '13':13, '20':20, '15':15, '10':10,
           '24':24, '22':22, '14':14, '21':21, '12':12, '23':23, '16':16, '26':26, '11':11, '9':9, '7':7,
           '19':19, '25':25, '6':6, '28':28, '17':17, '8':8, '三十四層':34, '(空白)':0, '三十五層':35, '三十九層':39,
           '三十六層':36, '四十二層':42, '三十八層':38, '四十三層':43, '5':5, '4':4, '34':34, '35':35, '29':29, '3':3,
           '37':37, '27':27, '32':32, '31':31, '18':18, '36':36, '八十五層':85, '四十一層':41, '三十七層':37, '六十八層':68,
           '30':30, '33':33, '四十層':40, '42':42, '一層':1, '五十層':50, '00Z':0, '00Y':0})

    extxl = extxl[extxl["總樓層數"]!=1]

    category_ = {
        '路名_型態':["縣市", '建物型態', '路名'],
        '同建物':["縣市", "建物型態", "總樓層數", "路名"],
        '精準':["縣市", "建物型態", "總樓層數", "路名", "屋齡2023"],
    }
    cate_out = {}

    for k in category_.keys():
        cate_out[k] = {
            'len':np.ones(df_all['ID'].shape[0]),
            'price':np.ones(df_all['ID'].shape[0]),
            'area':np.ones(df_all['ID'].shape[0]),
            'total_price':np.ones(df_all['ID'].shape[0]),
            'ID':np.ones(df_all['ID'].shape[0], dtype = 'object'),
                      }
    mask_list = ["縣市", '建物型態', "路名", "總樓層數", "屋齡2023"]
    j=0
    for district, group_df in df_all.groupby('鄉鎮市區'):

        g_extxl = extxl[extxl["鄉鎮市區"]==district]    
        group_df['移轉層次_cate'] = (group_df['移轉層次']>5)*1 + (group_df['移轉層次']>10)*1

        for i in range(group_df.shape[0]):

            mask_dict = {}

            for c in mask_list:
                if c=='路名':
                    mask_dict[c] = (g_extxl["土地位置建物門牌"].str.contains(group_df["路名"].iloc[i]))*1
                elif c=='屋齡2023':
                    mask_dict[c] = ((g_extxl["屋齡"]>=(group_df["屋齡"].iloc[i]))&(g_extxl["屋齡"]<=(group_df["屋齡"].iloc[i]+3))).astype('int')
                else:
                    mask_dict[c] = (g_extxl[c]==group_df[c].iloc[i])*1

            for name, map_cols in category_.items():

                mask = np.zeros(g_extxl.shape[0])

                for col in map_cols:
                    mask += mask_dict[col]

                gmdf = g_extxl[mask==len(map_cols)]

                cate_out[name]['len'][j] = gmdf.shape[0]
                cate_out[name]['price'][j] = gmdf['單價元平方公尺'].mean()
                cate_out[name]['total_price'][j] = gmdf['總價元'].mean()
                cate_out[name]['area'][j] = gmdf['建物移轉總面積平方公尺'].mean()

                cate_out[name]['ID'][j] = group_df['ID'].iloc[i]
            j+=1

    df_list = [(name, pd.DataFrame(dict_).fillna(-1)) for name, dict_ in cate_out.items()]

    for (name, df) in df_list:
        df.columns = [c+'_'+name+'2023' if c!='ID' else c for c in df.columns]

    total2023_df = pd.DataFrame({'ID':df_list[0][1]['ID']})

    for (name, df) in df_list:
        total2023_df = total2023_df.merge(df, on = 'ID')

    final_df = total_df.merge(total2020_df, on  = 'ID').merge(total2023_df, on = 'ID').merge(totalrev_df, on = 'ID')

    final_df.to_csv('ext_data_processed/extxl.csv', index = False)

if __name__ == '__main__':
    main()