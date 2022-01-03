# Task1: Calculate Average Rating based on current comments and compare it with the existing average rating
# Task2: Specify 20 reviews for the product that will be displayed on the product detail page.

# Görev1 Average Rating i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız
# Görev2 ürün için ürün detay sayfasında görüntülencek 20 review i belirleyiniz

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
df = pd.read_csv("hafta5/Ders Öncesi Notlar/amazon_review.csv")

df.head(10)
df.info()
df["overall"].mean()     #the average of the votes # oyların ortalaması  4.587589013224822

df["overall"].value_counts().sum()     # toplam 4915 oylanmmış # Voted 4915 times
df["overall"].value_counts()           # oyların dağılımı      # distribution of votes

# son tarih ve ilk tarihe bakalım   # Let's look at the last date and the first date
df["reviewTime"].max()       # 2014-12-07
df["reviewTime"].min()       # 2012-01-09

# It's not a very long-term product like 10 years, so let's
# divide our 2 years into 4 parts and get the rate of the last 6 months more.

# 10 sene gibi çok uzun soluklu ürünler değil o halde yaklaşık 2 senemizi
# 4 parçaya bölüp son 6 ayın oranını daha çok alalım

(df["day_diff"] < 180).sum()      # 581 people commented in the last 6 months # 581 tane kişi son 6 ay yorumlamış
((df["day_diff"] > 180) & (df["day_diff"] < 360)) .sum()      # 1358 people, 6-12 months # 1358 kişi 6-12 ay arası
((df["day_diff"] > 360) & (df["day_diff"] < 540)) .sum()      # 1319 people voted between 12-18 months # 1319 kişi 12-18 ay arasında oylamış
# 1657 people voted 18 months later, the first 6 months received less comments than the other 6 months
# 18 ay sonrasına da 1657 kişi kalıyor , ilk 6 ay diğer 6 aylara göre az yorum almış

# Let's see how many votes were used
# hangi oylardan ne kadar kullanılmış bakalım
first_6_months = df[(df["day_diff"] < 180)]
second_6_months = df[((df["day_diff"] > 180) & (df["day_diff"] < 360))]
third_6_months = df[((df["day_diff"] > 360) & (df["day_diff"] < 540))]
first_6_months["overall"].mean()     # 4.693631669535284

first_6_months["overall"].value_counts()      # 5p 475 people /4p 69 people /3p 16 /2p 7 /1p 14 people
second_6_months["overall"].value_counts()     # 1116         /147          /39    /37   /19
third_6_months["overall"].value_counts()      # 1058        /137          /66    /35   /23

# If we double the number of people giving points in the last 6 months, we will still reach the numbers between 6-18 months and 18-24 months
# In that case, we can say that there has been no change in the quality or other characteristics of the product.
# Maybe there was a lack of product promotion and customers saw less of the product
# or an alternative multi-purpose product came out.
# Still, if we want to calculate the average of the last 6 months, not equal, but 1 more;

# son 6 aydaki puan veren kişi sayıları nı yaklaşık 2 katını alırsak yine 6-18 ay ve 18-24 ay arasındaki
# sayılara ulaşırız
# o halde ürünün kalitesinde veya başka özelliğinde değişim olmamış diyebiliriz
# belki ürün tanıtımında eksiklik oldu ve müşteriler ürünü daha az gördü
# ya da alternatif çok amaçlı bir ürün çıktı
# yine de son oranları eşit derecede değilde son 6 ayı 1 fazla olcak şekilde ortalamasını hesaplamak istersek;


df.loc[df["day_diff"] <= 180, "overall"].mean() * 28 / 100 + \
df.loc[(df["day_diff"] > 180) & (df["day_diff"] <= 360), "overall"].mean() * 26 / 100 + \
df.loc[(df["day_diff"] > 360) & (df["day_diff"] <= 540), "overall"].mean() * 24 / 100 + \
df.loc[(df["day_diff"] > 540), "overall"].mean() * 22 / 100

# Out[29]: 4.616186142066258            # As you can see, there is not much difference.# görüldüğü üzere çok fark yok

# Let's functionalize it and make it more organized
# Fonksiyonlaştırıp daha düzenli hale getirelim

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df,34,22,22,22)     # Out[32]: 4.698145885727909

# For review, it is necessary to be able to present the comment that the customer is looking for objectively.
# To the left of the helpful column is a vote of like, and to the right of it is a vote of dislike.
# we also have the helpful_yes number of likes.
# subtracting negatives from positives or making positive/(pos + neg) to keep the best scores is the incomplete solution.
# best of all, wilson lower bound score with confidence interval

# review için
# müşteriye objektif bir şekilde aradığı yorumu sunabilmek lazım
# helpful sütünun solundaki faydalı bulma sağındakiler faydasız bulma oyudur
# elimizde birde helpful_yes yanı begenılme sayımız var
# en iyi score ların elimizde kalması için pozitiflerden negatifleri çıkarmak
# veya pozıtıf/(poz + neg) yapmak yanlış çözüm en doğrusu güven aralığı içeren wilson lower bound score


df["helpful"].head(50)     # dtype object
df["helpful_yes"].head(20)     # dtype int64
# total vote(değerlendirmeye verilen oy sayısı) olumluları çıkarırsak olumsuzlar kalır
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df["helpful_no"].head()
df["total_vote"].head()

# let's create a dataframe
# dataframe oluşturalım
comments = pd.DataFrame({"up": df["helpful_yes"], "down": df["helpful_no"]})

# with a function that gives the top 20 product suggestions
# en iyi 20 ürün önerisini veren fonksiyon ile
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"],x["down"]),axis=1)

# we have listed our products
# ürünlerimizi sıralamış olduk
comments.sort_values("wilson_lower_bound", ascending=False).head(20)