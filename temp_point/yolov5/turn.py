from pymongo import MongoClient
from read_thermal import flir_image_extractor
client = MongoClient('mongodb://140.118.172.141:27017/')
db = client.TSMC_test
# collection2 = db.if_detect

# data = {"number" : 1}
# collection2.insert_one(data)
collection3 = db.Unidentified

image_data = collection3.find_one({"processed" : False})
path_list = image_data["image"]
dir_name = image_data["time"]
path_base = "C:\\Users\\ki\\Desktop\\python\\vue-V8\\vue-V8\\"

img_names = []

for i in range(len(path_list)):
    img_names.append(path_base + path_list[i])

fir = flir_image_extractor.FlirImageExtractor()
for img_name in img_names:
    print("==")
    fir.process_image(str(img_name))
    fir.save_images(dir_name)#原圖轉成可視影像和熱影像(無其他字擋住


query = {"_id" : image_data["_id"]}
update = {"$set": {"processed": True}}
collection3.update_one(query, update)