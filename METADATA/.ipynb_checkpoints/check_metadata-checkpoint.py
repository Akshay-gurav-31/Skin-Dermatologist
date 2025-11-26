import pandas as pd

# Metadata files load करा
meta = pd.read_csv(r"D:\SKIN DECEASE\METADATA\Skin_Metadata.csv")
train_split = pd.read_csv(r"D:\SKIN DECEASE\METADATA\train_split.csv")
test_split = pd.read_csv(r"D:\SKIN DECEASE\METADATA\test_split.csv")


print("✅ Files Loaded Successfully!\n")

print("📊 Skin_Metadata.csv sample:")
print(meta.head(), "\n")

print("📈 Labels Count:")
print(meta['label'].value_counts(), "\n")

print("🧩 Train Split Size:", len(train_split)) 
print("🧪 Test Split Size:", len(test_split))
