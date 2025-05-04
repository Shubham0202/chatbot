from pymongo import MongoClient

def get_apartment_data():
    try:
        # Connect to local MongoDB server (adjust URI if needed)
        client = MongoClient("mongodb://localhost:27017/")
        
        # Select the database and collection
        db = client["apartment_chatbot"]
        collection = db["apartments"]
        
        # Fetch all documents
        data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB's _id field
        return data

    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return []
