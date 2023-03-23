##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.
                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# BONUS: Tüm süreci fonksiyonlaştırınız.


###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################
#!pip install lifetimes
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

df_ = pd.read_csv(r"C:/Users/cemil/OneDrive/Masaüstü/VBO/kurs/crmAnalytics-221211-020816/FLOCLTVPrediction-230305-185416/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()
df.head()
#                              master_id order_channel last_order_channel first_order_date last_order_date last_order_date_online last_order_date_offline  order_num_total_ever_online  order_num_total_ever_offline  customer_value_total_ever_offline  customer_value_total_ever_online       interested_in_categories_12  order_num_total  customer_value_total
#0  cc294636-19f0-11eb-8d74-000d3a38a36f   Android App            Offline       2020-10-30      2021-02-26             2021-02-21              2021-02-26                         4.00                          1.00                             139.99                            799.38                           [KADIN]             5.00                939.37
#1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f   Android App             Mobile       2017-02-08      2021-02-16             2021-02-16              2020-01-10                        19.00                          2.00                             159.97                           1853.58  [ERKEK, COCUK, KADIN, AKTIFSPOR]            21.00               2013.55
#2  69b69676-1a40-11ea-941b-000d3a38a36f   Android App        Android App       2019-11-27      2020-11-27             2020-11-27              2019-12-01                         3.00                          2.00                             189.97                            395.35                    [ERKEK, KADIN]             5.00                585.32
#3  1854e56c-491f-11eb-806e-000d3a38a36f   Android App        Android App       2021-01-06      2021-01-17             2021-01-17              2021-01-06                         1.00                          1.00                              39.99                             81.98               [AKTIFCOCUK, COCUK]             2.00                121.97
#4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       Desktop            Desktop       2019-08-03      2021-03-07             2021-03-07              2019-08-03                         1.00                          1.00                              49.99                            159.99                       [AKTIFSPOR]             2.00                209.98
# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()

#                            customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg
#0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     30.57       5.00             187.87
#1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    224.86      21.00              95.88
#2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     78.86       5.00             117.06
#3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     20.86       2.00              60.98
#4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     95.43       2.00             104.99

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly']
        )
bgf
#<lifetimes.BetaGeoFitter: fitted with 19945 subjects, a: 0.00, alpha: 76.17, b: 0.00, r: 3.66>

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])
# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]
#                                customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month
#7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f                62.71     67.29      52.00             166.22               4.66               9.31
#15611  4a7e875e-e6ce-11ea-8f44-000d3a38a36f                39.71     40.00      29.00             165.30               3.37               6.75
#8328   1902bf80-0035-11eb-8341-000d3a38a36f                28.86     33.29      25.00              97.44               3.14               6.28
#19538  55d54d9e-8ac7-11ea-8ec0-000d3a38a36f                52.57     58.71      31.00             228.53               3.08               6.17
#14373  f00ad516-c4f4-11ea-98f7-000d3a38a36f                38.00     46.43      27.00             141.35               3.00               6.00
#10489  7af5cd16-b100-11e9-9757-000d3a38a36f               103.14    111.86      43.00             157.11               2.98               5.96
#4315   d5ef8058-a5c6-11e9-a2fc-000d3a38a36f               133.14    147.14      49.00             161.85               2.83               5.66
#6756   27310582-6362-11ea-a6dc-000d3a38a36f                62.71     64.14      29.00             168.88               2.79               5.59
#6666   53fe00d4-7b7a-11eb-960b-000d3a38a36f                 9.71     13.00      17.00             259.87               2.78               5.56
#10536  e143b6fa-d6f8-11e9-93bc-000d3a38a36f               104.57    113.43      40.00             176.20               2.76               5.53


cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]
#                                customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month
#7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f                62.71     67.29      52.00             166.22               4.66               9.31
#15611  4a7e875e-e6ce-11ea-8f44-000d3a38a36f                39.71     40.00      29.00             165.30               3.37               6.75
#8328   1902bf80-0035-11eb-8341-000d3a38a36f                28.86     33.29      25.00              97.44               3.14               6.28
#19538  55d54d9e-8ac7-11ea-8ec0-000d3a38a36f                52.57     58.71      31.00             228.53               3.08               6.17
#14373  f00ad516-c4f4-11ea-98f7-000d3a38a36f                38.00     46.43      27.00             141.35               3.00               6.00
#10489  7af5cd16-b100-11e9-9757-000d3a38a36f               103.14    111.86      43.00             157.11               2.98               5.96
#4315   d5ef8058-a5c6-11e9-a2fc-000d3a38a36f               133.14    147.14      49.00             161.85               2.83               5.66
#6756   27310582-6362-11ea-a6dc-000d3a38a36f                62.71     64.14      29.00             168.88               2.79               5.59
#6666   53fe00d4-7b7a-11eb-960b-000d3a38a36f                 9.71     13.00      17.00             259.87               2.78               5.56
#10536  e143b6fa-d6f8-11e9-93bc-000d3a38a36f               104.57    113.43      40.00             176.20               2.76               5.53


# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()
#                            customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  exp_average_value
#0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     30.57       5.00             187.87               0.97               1.95             193.63
#1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    224.86      21.00              95.88               0.98               1.97              96.67
#2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     78.86       5.00             117.06               0.67               1.34             120.97
#3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     20.86       2.00              60.98               0.70               1.40              67.32
#4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     95.43       2.00             104.99               0.40               0.79             114.33

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv.head()
#0   395.73
#1   199.43
#2   170.22
#3    98.95
#4    95.01
#Name: clv, dtype: float64

cltv_df["cltv"]  = cltv

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("cltv",ascending=False)[:20]

#                                customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  exp_average_value    cltv
#9055   47a642fe-975b-11eb-8c2a-000d3a38a36f                 2.86      7.86       4.00            1401.80               1.09               2.19            1449.06 3327.78
#13880  7137a5c0-7aad-11ea-8f20-000d3a38a36f                 6.14     13.14      11.00             758.09               1.97               3.94             767.36 3172.39
#17323  f59053e2-a503-11e9-a2fc-000d3a38a36f                51.71    101.00       7.00            1106.47               0.72               1.44            1127.61 1708.98
#12438  625f40a2-5bd2-11ea-98b0-000d3a38a36f                74.29     74.57      16.00             501.87               1.57               3.13             506.17 1662.61
#7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f                62.71     67.29      52.00             166.22               4.66               9.31             166.71 1628.89
#8868   9ce6e520-89b0-11ea-a6e7-000d3a38a36f                 3.43     34.43       8.00             601.23               1.27               2.53             611.49 1623.81
#6402   851de3b4-8f0c-11eb-8cb8-000d3a38a36f                 8.29      9.43       2.00             862.69               0.79               1.59             923.68 1538.86
#6666   53fe00d4-7b7a-11eb-960b-000d3a38a36f                 9.71     13.00      17.00             259.87               2.78               5.56             262.07 1529.23
#19538  55d54d9e-8ac7-11ea-8ec0-000d3a38a36f                52.57     58.71      31.00             228.53               3.08               6.17             229.61 1485.82
#14858  031b2954-6d28-11eb-99c4-000d3a38a36f                14.86     15.57       3.00             743.59               0.87               1.74             778.05 1423.00
#17963  8fd88976-6708-11ea-9d38-000d3a38a36f                50.29     63.29       7.00             694.20               0.92               1.84             707.69 1362.61
#15516  9083981a-f59e-11e9-841e-000d3a38a36f                63.57     83.86       4.00            1090.36               0.57               1.15            1127.35 1359.44
#6717   40b4f318-9dfb-11eb-9c47-000d3a38a36f                27.14     33.86       7.00             544.70               1.16               2.33             555.41 1355.44
#4157   7eed6468-4540-11ea-acaf-000d3a38a36f                89.14     90.00      27.00             289.76               2.21               4.43             291.29 1353.53
#4735   dbabb58e-6312-11ea-a6dc-000d3a38a36f                61.29     64.29      13.00             442.12               1.42               2.85             446.82 1334.83
#11694  90f1b7f2-bbad-11ea-a0c9-000d3a38a36f                47.29     48.00       6.00             647.34               0.93               1.87             662.11 1297.52
#11179  d2e74a36-3228-11eb-860c-000d3a38a36f                 1.14     26.29       3.00             750.57               0.78               1.56             785.34 1286.14
#1853   f02473b0-43c3-11eb-806e-000d3a38a36f                17.29     23.14       2.00             835.88               0.68               1.37             895.04 1285.23
#5775   e31293ac-d63a-11e9-93bc-000d3a38a36f                91.71     93.14       8.00             727.09               0.83               1.65             739.39 1282.58
#7312   90befc98-925a-11eb-b584-000d3a38a36f                 4.14      8.86       6.00             431.33               1.36               2.73             441.40 1263.19

###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

#                            customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  exp_average_value   cltv cltv_segment
#0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     30.57       5.00             187.87               0.97               1.95             193.63 395.73            A
#1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    224.86      21.00              95.88               0.98               1.97              96.67 199.43            B
#2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     78.86       5.00             117.06               0.67               1.34             120.97 170.22            B
#3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     20.86       2.00              60.98               0.70               1.40              67.32  98.95            D
#4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     95.43       2.00             104.99               0.40               0.79             114.33  95.01            D



# Tüm işlemin Fonksiyonlaştırılması.


def create_cltv_df(dataframe):

    # Veriyi Hazırlama
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # CLTV veri yapısının oluşturulması
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # # Gamma-Gamma Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # Cltv tahmini
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentleme
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)


cltv_df.head(10)
#                            customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  exp_average_value   cltv cltv_segment
#0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     30.57       5.00             187.87               0.97               1.95             193.63 395.73            A
#1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    224.86      21.00              95.88               0.98               1.97              96.67 199.43            B
#2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     78.86       5.00             117.06               0.67               1.34             120.97 170.22            B
#3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     20.86       2.00              60.98               0.70               1.40              67.32  98.95            D
#4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     95.43       2.00             104.99               0.40               0.79             114.33  95.01            D
#5  e585280e-aae1-11e9-a2fc-000d3a38a36f               120.86    132.29       3.00              66.95               0.38               0.77              71.35  57.43            D
#6  c445e4ee-6242-11ea-9d1a-000d3a38a36f                32.57     64.86       4.00              93.98               0.65               1.30              98.13 134.28            C
#7  3f1b4dc8-8a7d-11ea-8ec0-000d3a38a36f                12.71     54.57       2.00              81.81               0.52               1.04              89.57  97.70            D
#8  cfbda69e-5b4f-11ea-aca7-000d3a38a36f                58.43     70.71       5.00             210.94               0.71               1.42             217.30 322.73            A
#9  1143f032-440d-11ea-8b43-000d3a38a36f                61.71     96.00       2.00              82.98               0.39               0.79              90.81  75.22            D








