# Tbrain 永豐AI GO競賽-攻房戰

## 隊伍名稱 : 台南GOGO

---

## Description

- 0.real_price_dataset_preproc.py : 針對實價登錄資料進行前處理
	- 預設資料放置路徑: ./external_data/實價登錄/...
- 1.Extra_feature.py : 將外部資料進行處理並轉換為建模時需要的特徵
	- 處理完的檔案會存於./ext_data_processed
- 2.Preproc_modeling.py : 資料前處理、其他特徵工程、模型訓練與預測

## Running

1. clone repository
2. 下載實價登錄資料並解壓縮至./external資料夾中
	- [本次競賽使用的所有實價登錄資料](https://drive.google.com/file/d/1MiKuqADlzohEteiMsTDi8ZhA2JDO9moq/view?usp=sharing)
3. 安裝套件
```bash
pip install -r requirements.txt
```

4. 外部資料處理與額外特徵檔案產生
```bash
# 實價登錄檔案合併
python 0.real_price_dataset_preproc.py

# 外部資料處理與特徵轉換
python 1.Extra_feature.py
```

5. 資料前處理與模型訓練

```bash
# 驗證模式
python 2.Preproc_modeling.py --mode validation

# 預測模式
python 2.Preproc_modeling.py --mode test
```