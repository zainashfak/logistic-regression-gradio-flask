import urllib.request

base = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/"
urllib.request.urlretrieve(base + "ISLR/Default.csv", "data/dataset1.csv")
urllib.request.urlretrieve(base + "MASS/biopsy.csv", "data/dataset2.csv")
print("Datasets downloaded")