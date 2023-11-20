import os
import re
import glob
import math
import ntpath
import pandas as pd
import numpy as np
import calendar


mapping_dict = {'a':'台北市',
                'b':'臺中市',
                'c':'基隆市',
                'd':'台南市',
                'e':'高雄市',
                'f':'新北市',
                'g':'宜蘭縣',
                'h':'桃園市',
                'j':'新竹縣',
                'k':'苗栗縣',
                'm':'南投縣',
                'n':'彰化縣',
                'p':'雲林縣',
                'q':'嘉義縣',
                't':'屏東縣',
                'u':'花蓮縣',
                'v':'台東縣',
                'x':'澎湖縣',
                'w':'金門縣',
                'z':'連江縣',
                'i':'嘉義市',
                'o':'新竹市'}

def chi2int_number(num_str):
    chi_to_int_table = pd.DataFrame({
        'chi': ['一', '二', '三', '四', '五', '六', '七', '八', '九', '全', '十'],
        'int': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '全', '10']
    })
    
    num_str = str(num_str)
    num_str = num_str.replace('層', '')
    if num_str[0] in chi_to_int_table['int'].values:
        return num_str
    else:
        if '，' in num_str or '地下' in num_str or num_str == 'nan' or num_str[0] not in chi_to_int_table['chi'].values or ' ' in num_str:
            print(f'格式太奇怪：{num_str} -> 0')
            return '0'

        else:
            if len(num_str) == 1:
                if num_str[0] == '十':  # 十 -> 10
                    return(num_str.replace('十', '10'))
                elif num_str[0] == '全':
                    return(chi_to_int_table[chi_to_int_table['chi'] == num_str]['int'].values[0])
                else:  # 一到九 -> 1-9
                    return(chi_to_int_table[chi_to_int_table['chi'] == num_str]['int'].values[0])

            elif len(num_str) == 2:
                if num_str[0] == '十':  # 十一到十九 -> 11-19
                    num_int = num_str.replace('十', '1')
                    num_int = num_int.replace(num_str[1], chi_to_int_table[chi_to_int_table['chi'] == num_str[1]]['int'].values[0])
                    return(num_int)

                elif num_str[0] != '十' and num_str[1] == '十':  # 三十、四十、五十... -> 30, 40, 50, ...
                    num_int = num_str.replace(num_str[0], chi_to_int_table[chi_to_int_table['chi'] == num_str[0]]['int'].values[0])
                    num_int = num_int.replace('十', '0')
                    return(num_int)

            elif len(num_str) == 3:  # 二十一... -> 21
                num_int = num_str.replace(num_str[0], chi_to_int_table[chi_to_int_table['chi'] == num_str[0]]['int'].values[0])
                num_int = num_int.replace('十', '')
                if num_str[0] != num_str[2]:
                    num_int = num_int.replace(num_str[2], chi_to_int_table[chi_to_int_table['chi'] == num_str[2]]['int'].values[0])
                return(num_int)

            else:
                print(f'格式太奇怪：{num_str} -> 0')
                return '0'

def todatetime(string):
    # 只分成7碼、6碼、4碼
    if (isinstance(string, int) or isinstance(string, float)) and np.isnan(string):
        return np.datetime64('NaT')
    
    elif isinstance(string, str) and (' ' in string or '-' in string):
        return np.datetime64('NaT')
    
    else:
        string = str(int(string))
        
        if len(string) == 4:
            year = str(int(string[:2]) + 1911)
            month = '01' if string[2:] == '00' else string[2:]
            day = '01'
            
        elif len(string) == 6:
            string = '0' + str(string)
            year = str(int(string[:3]) + 1911)
            month = string[3:5]
            day = string[5:8]
            
        elif len(string) == 7:
            year = str(int(string[:3]) + 1911)
            month = string[3:5]
            day = string[5:8]
            
        else:
            print(f'{string} -> "NaT"')
            return np.datetime64('NaT')
                
        year = '2023' if int(year) > 2023 else year
        month = '01' if (month == '00' or int(month) > 12) else month
        day = '01' if day == '00' else day
        day = str(calendar.monthrange(int(year), int(month))[1]) if int(day) > calendar.monthrange(int(year), int(month))[1] else day
                
        print(f'{string} -> {year}/{month}/{day}')
        return  year + '/' + month + '/' + day
        

def find_other_data(sample, city, sale_type):
    global df
    
    ID = sample['編號']
    if sale_type == '不動產':
        age = np.nan if df['建物'][df['建物']['編號'] == ID]['屋齡'].shape[0] == 0 else df['建物'][df['建物']['編號'] == ID]['屋齡'].mean()
    elif sale_type == '預售屋':
        age = 0
    park_number = np.nan if '停車位' not in list(df.keys()) else df['停車位'][df['停車位']['編號'] == ID].shape[0]

    return age, park_number, city, f'{sale_type}買賣'

if __name__ == '__main__':
    
    saved_dir = './ext_data_processed/'
    main_dir = './external_data/實價登錄資料'
    
    # 實價登錄 - 2021/2022
    file_idx = 0

    for sale_code, sale_type in zip(['a', 'b'], ['不動產', '預售屋']):
        for time_intvl in sorted(os.listdir(main_dir)):
            if (time_intvl[:4]!='2021')&(time_intvl[:4]!='2022'):
                continue
            if time_intvl != '.DS_Store':
                for xls_file in sorted([
                    os.path.join(main_dir, time_intvl, f) for f in os.listdir(os.path.join(main_dir, time_intvl)) if re.search('lvr_land_'+sale_code, f)
                ]):
                    manifest = pd.read_csv(os.path.join(main_dir, time_intvl, 'manifest.csv'))
                    city = manifest[manifest['name'] == ntpath.split(xls_file)[-1]]['description'].values[0][:3]

                    if file_idx == 0:
                        print(xls_file, end=f', {city}{sale_type}...')
                        df = pd.read_excel(xls_file, sheet_name=None, header=0, skiprows=[1])
                        if not f"{sale_type}買賣" in df.keys():  # 沒有這個表格，跳過
                            print(f'sheet did not exist, skipped.')
                            continue

                        df[f"{sale_type}買賣"] = df[f"{sale_type}買賣"][df[f"{sale_type}買賣"]['交易標的'].str.contains('房地', regex=False)]
                        print(f'{df[f"{sale_type}買賣"].shape[0]} samples.')
                        if df[f"{sale_type}買賣"].shape[0] == 0:  # 樣本數為0，跳過
                            print(f'number of samples = 0, skipped.')
                            continue

                        df[f"{sale_type}買賣"][['屋齡', '車位個數', '縣市', '買賣類別']] = df[f"{sale_type}買賣"].apply(find_other_data, axis=1, args=(city, sale_type), result_type='expand')
                        df_all = df[f"{sale_type}買賣"].copy()

                        file_idx += 1
                    else:
                        print(xls_file, end=f', {city}{sale_type}...')
                        df = pd.read_excel(xls_file, sheet_name=None, header=0, skiprows=[1])
                        if not f"{sale_type}買賣" in df.keys():  # 沒有這個表格，跳過
                            print(f'sheet did not exist, skipped.')
                            continue

                        df[f"{sale_type}買賣"] = df[f"{sale_type}買賣"][df[f"{sale_type}買賣"]['交易標的'].str.contains('房地', regex=False)]
                        print(f'{df[f"{sale_type}買賣"].shape[0]} samples.')
                        if df[f"{sale_type}買賣"].shape[0] == 0:  # 樣本數為0，跳過
                            print(f'number of samples = 0, skipped.')
                            continue

                        df[f"{sale_type}買賣"][['屋齡', '車位個數', '縣市', '買賣類別']] = df[f"{sale_type}買賣"].apply(find_other_data, axis=1, args=(city, sale_type), result_type='expand')
                        df_all = pd.concat([df_all, df[f"{sale_type}買賣"]], ignore_index=True)

                        file_idx += 1
    df_all = df_all.reindex(columns=(['縣市'] + list([a for a in df_all.columns if a != '縣市'])))

    df_all['縣市'] = df_all['縣市'].str.replace('臺', '台')
    df_all['移轉層次_數字'] = df_all['移轉層次'].apply(chi2int_number)
    df_all['總樓層數_數字'] = df_all['總樓層數'].apply(chi2int_number)
    df_all['建築完成年月'] = pd.to_datetime(df_all['建築完成年月'].apply(todatetime))
    df_all['交易年月日'] = pd.to_datetime(df_all['交易年月日'].apply(todatetime))

    df_all.to_csv(os.path.join(saved_dir, '實價登錄資料_處理後_2021-2022.csv'), index=False)
    
    
    
    # 實價登錄 - 2020 (尚未清理資料)
    files_list = []
    
    for sub_dir in ['2020-1', '2020-2', '2020-3', '2020-4']:
        f = [os.path.join(main_dir, sub_dir, f) for f in os.listdir(os.path.join(main_dir, sub_dir)) if re.search('_[a-z]{1}.csv', f)]
        files_list = files_list + f

    df_list = []

    for path in files_list:
        df = pd.read_csv(path)
        df['縣市'] = mapping_dict[ntpath.split(path)[-1][0]]
        df_list.append(df)
    
    df_all = pd.concat(df_list)
    df_all.to_csv(os.path.join(saved_dir, '實價登錄資料_2020全年.csv'), index = False)
    

    # 實價登錄 - 2023 (尚未清理資料)
    files_list = []
    
    for sub_dir in ['2023-1', '2023-2', '2023-3']:
        f = [os.path.join(main_dir, sub_dir, f) for f in os.listdir(os.path.join(main_dir, sub_dir)) if re.search('_[a-z]{1}.csv', f)]
        files_list = files_list + f

    df_list = []

    for path in files_list:
        df = pd.read_csv(path)
        df['縣市'] = mapping_dict[ntpath.split(path)[-1][0]]
        df_list.append(df)
    
    df_all = pd.concat(df_list)
    df_all.to_csv(os.path.join(saved_dir, '實價登錄資料_2023q1q3.csv'), index = False)