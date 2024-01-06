from pymongo import MongoClient
client = MongoClient('mongodb://140.118.172.141:27017/')
db = client.TSMC_test
collection2 = db.if_detect

#data = {"number" : 0}
result = collection2.find_one()
print(result["_id"])

query = {"_id" : result["_id"]}
update = {"$set": {"number": 1}}

collection2.update_one(query, update)