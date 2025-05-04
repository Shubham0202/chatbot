from .data_loader import handle_user_query

def get_chat_response(query):
    matches = handle_user_query(query)

    if not matches:
        return {
            "status": "no_results",
            "message": "Sorry, no matching apartments found.",
            "results": []
        }

    # If the response is a greeting, return it directly
    if matches[0].get("type") == "greeting":
        return {
            "status": "greeting",
            "message": matches[0]["summary"],
            "results": []
        }

    results = []
    for match in matches:
        results.append({
            "summary": match.get("summary", "No summary available."),
            "similarity_score": round(float(match.get("similarity_score", 0)), 3)
        })

    return {
        "status": "success",
        "query": query,
        "results": results
    }
