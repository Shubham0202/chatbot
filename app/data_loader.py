from pymongo import MongoClient
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import re

# === Connect to MongoDB ===
client = MongoClient("mongodb://localhost:27017/")
db = client["apartment_chatbot"]
collection = db["apartments"]

# === Get all unique locations from database ===
def get_unique_locations():
    pipeline = [
        {"$group": {"_id": "$location"}},
        {"$project": {"location": "$_id", "_id": 0}}
    ]
    locations = list(collection.aggregate(pipeline))
    return [loc["location"] for loc in locations if loc.get("location")]

# === Get apartment listings ===
def get_apartment_data():
    return list(collection.find({}, {"_id": 0}))

# === Convert MongoDB data to LangChain Documents ===
def create_langchain_documents(data):
    documents = []
    for apartment in data:
        amenities = ", ".join(apartment.get('amenities', []))
        location = apartment.get('location', 'an unknown location')
        price = apartment.get('price', 'a negotiable price')
        bedrooms = apartment.get('bedrooms', '?')
        area = apartment.get('area_sqft', 'unknown area')

        text = (
            f"This is a {bedrooms} BHK apartment in {location}. "
            f"It spans {area} sqft and is priced at ₹{price}. "
            f"Key amenities include: {amenities}."
        )

        metadata = {
            "bedrooms": bedrooms,
            "location": location,
            "price": price,
            "area_sqft": area,
            "amenities": amenities,
            "city": location.split(",")[-1].strip().lower() if location else "unknown"
        }

        documents.append(Document(page_content=text, metadata=metadata))
    return documents

# === Initialize embedding model ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Create ChromaDB vector store ===
persist_dir = "./chroma_apartments"
os.makedirs(persist_dir, exist_ok=True)
apartment_data = get_apartment_data()
documents = create_langchain_documents(apartment_data)
vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=persist_dir)

# === Dynamic Location Extraction ===
def extract_location_from_query(query, available_locations):
    query = query.lower()
    # Create a mapping of location names to their standardized forms
    location_map = {loc.lower(): loc for loc in available_locations}
    
    # Check for direct matches first
    for loc_lower, loc_standard in location_map.items():
        if loc_lower in query:
            return loc_standard
    
    # Check for partial matches (city names without area)
    for loc_lower, loc_standard in location_map.items():
        city_part = loc_lower.split(",")[0].strip()
        if city_part in query:
            return loc_standard
    
    # Check for known city aliases
    city_aliases = {
        "pune": ["pune", "puna"],
        "mumbai": ["mumbai", "bombay"],
        "bangalore": ["bangalore", "bengaluru"],
        "delhi": ["delhi", "new delhi", "dilli"]
    }
    
    for alias_list in city_aliases.values():
        for alias in alias_list:
            if alias in query:
                # Find the first location that contains this city
                for loc_lower, loc_standard in location_map.items():
                    if alias in loc_lower:
                        return loc_standard
    return None

# === Enhanced Query Intent Classification ===
def classify_user_query(query: str, available_locations: list) -> str:
    query = query.lower().strip()
    greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
    
    # Check for greetings first
    if any(greet in query for greet in greetings):
        return "greeting"
    
    # Extract location if present
    location = extract_location_from_query(query, available_locations)
    
    # Check for sorting queries with location specifications
    if re.search(r"(most expensive|highest price|most costly)", query):
        return "sort_by_price_desc_location" if location else "sort_by_price_desc"
    elif re.search(r"(cheapest|lowest price|most affordable)", query):
        return "sort_by_price_asc_location" if location else "sort_by_price_asc"
    elif re.search(r"(biggest|largest)", query):
        return "sort_by_area_desc_location" if location else "sort_by_area_desc"
    elif re.search(r"(smallest|tiniest)", query):
        return "sort_by_area_asc_location" if location else "sort_by_area_asc"
    
    # Check for filter-based queries
    if any(word in query for word in ["with", "in", "under", "above", "near", "bhk", "bedroom", "budget"]):
        return "filter_search"
    
    # Default to semantic search
    return "semantic_search"

# === Enhanced Filter-based Search ===
def filter_based_search(query, available_locations):
    query = query.lower()
    filter_ = {}
    
    # Extract BHK information
    bhk_match = re.search(r'(\d)\s?bhk', query)
    if bhk_match:
        filter_["bedrooms"] = int(bhk_match.group(1))
    
    # Extract location
    location = extract_location_from_query(query, available_locations)
    if location:
        filter_["location"] = location
    
    # Extract price range
    if "under" in query or "below" in query:
        try:
            amount = int(re.search(r'(under|below)\s?(\d+)', query).group(2))
            filter_["price"] = {"$lte": amount}
        except:
            pass
    elif "above" in query or "over" in query:
        try:
            amount = int(re.search(r'(above|over)\s?(\d+)', query).group(2))
            filter_["price"] = {"$gte": amount}
        except:
            pass
    
    # Extract amenities
    amenities = []
    if "pool" in query:
        amenities.append("Swimming Pool")
    if "gym" in query:
        amenities.append("Gym")
    if "parking" in query:
        amenities.append("Parking")
    if amenities:
        filter_["amenities"] = {"$all": amenities}
    
    return list(collection.find(filter_, {"_id": 0}))

# === Enhanced ChromaDB Semantic Search ===
def chromadb_semantic_search(query, top_k=5):
    results = vectorstore.similarity_search_with_score(query, k=top_k*2)  # Get more results to filter
    unique = set()
    filtered = []
    
    for doc, score in results:
        meta = doc.metadata
        key = (meta['bedrooms'], meta['location'], meta['price'], meta['area_sqft'])
        
        if key not in unique:
            meta["similarity_score"] = round(float(score), 4)
            meta["summary"] = (
                f"🏠 A {meta['bedrooms']} BHK apartment in {meta['location']} spanning {meta['area_sqft']} sqft, "
                f"available at ₹{meta['price']}. Amenities include {meta['amenities']}."
            )
            filtered.append(meta)
            unique.add(key)
        
        if len(filtered) >= top_k:
            break
    
    return filtered

# === Handle Query with Location-Specific Sorting ===
def handle_location_specific_sort(query, sort_field, sort_order, available_locations, top_k=5):
    location = extract_location_from_query(query, available_locations)
    filter_ = {}
    if location:
        filter_["location"] = location
    
    results = list(collection.find(filter_, {"_id": 0}).sort(sort_field, sort_order).limit(top_k))
    return results

# === Main Query Handler ===
def handle_user_query(query: str, top_k=5):
    # Get available locations from database
    available_locations = get_unique_locations()
    
    # Classify query intent
    intent = classify_user_query(query, available_locations)

    if intent == "greeting":
        return [{
            "summary": "👋 Hello! I'm your apartment search assistant. How can I help you find your perfect home today?",
            "type": "greeting"
        }]

    # Handle location-specific sorting
    if intent == "sort_by_price_desc_location":
        results = handle_location_specific_sort(query, "price", -1, available_locations, top_k)
    elif intent == "sort_by_price_asc_location":
        results = handle_location_specific_sort(query, "price", 1, available_locations, top_k)
    elif intent == "sort_by_area_desc_location":
        results = handle_location_specific_sort(query, "area_sqft", -1, available_locations, top_k)
    elif intent == "sort_by_area_asc_location":
        results = handle_location_specific_sort(query, "area_sqft", 1, available_locations, top_k)
    # Handle general sorting
    elif intent == "sort_by_price_desc":
        results = list(collection.find({}, {"_id": 0}).sort("price", -1).limit(top_k))
    elif intent == "sort_by_price_asc":
        results = list(collection.find({}, {"_id": 0}).sort("price", 1).limit(top_k))
    elif intent == "sort_by_area_desc":
        results = list(collection.find({}, {"_id": 0}).sort("area_sqft", -1).limit(top_k))
    elif intent == "sort_by_area_asc":
        results = list(collection.find({}, {"_id": 0}).sort("area_sqft", 1).limit(top_k))
    # Handle filter search
    elif intent == "filter_search":
        results = filter_based_search(query, available_locations)
    # Default to semantic search
    else:
        results = chromadb_semantic_search(query, top_k)

    # Enrich results with better formatting
    enriched_results = []
    for item in results:
        summary = (
            f"🏠 **{item.get('bedrooms', '?')} BHK Apartment**\n"
            f"📍 **Location:** {item.get('location', 'unknown location')}\n"
            f"📏 **Area:** {item.get('area_sqft', '?')} sqft\n"
            f"💰 **Price:** ₹{item.get('price', '?')}\n"
            f"🛁 **Amenities:** {', '.join(item.get('amenities', [])) if isinstance(item.get('amenities'), list) else item.get('amenities', 'None')}\n"
        )
        
        if "similarity_score" in item:
            summary += f"🔍 **Relevance Score:** {item['similarity_score']:.2f}"
        
        enriched_item = {
            "summary": summary,
            "details": {
                "bedrooms": item.get('bedrooms'),
                "location": item.get('location'),
                "price": item.get('price'),
                "area_sqft": item.get('area_sqft'),
                "amenities": item.get('amenities')
            }
        }
        enriched_results.append(enriched_item)

    # Add context message if no results found
    if not enriched_results:
        return [{
            "summary": "🔍 I couldn't find any apartments matching your criteria. Try broadening your search or adjusting your filters.",
            "type": "no_results"
        }]
    
    # Add header message based on intent
    header = None
    location = extract_location_from_query(query, available_locations)
    
    if "sort_by_price_desc" in intent:
        header = f"Here are the most expensive apartments{f' in {location}' if location else ' we have'}:"
    elif "sort_by_price_asc" in intent:
        header = f"Here are the most affordable apartments{f' in {location}' if location else ' we have'}:"
    elif "sort_by_area_desc" in intent:
        header = f"Here are the largest apartments{f' in {location}' if location else ' we have'}:"
    elif "sort_by_area_asc" in intent:
        header = f"Here are the most compact apartments{f' in {location}' if location else ' we have'}:"
    elif intent == "filter_search":
        header = "Here are apartments matching your filters:"
    else:
        header = "Here are some apartments that might interest you:"
    
    if header:
        enriched_results.insert(0, {
            "summary": f"✨ {header}",
            "type": "header"
        })
    
    return enriched_results