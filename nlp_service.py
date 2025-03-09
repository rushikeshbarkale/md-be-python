import logging

from flask import Flask, request, jsonify
from product_fetcher import extract_entities, fetch_products, preprocess_query, fetch_medical_equipment_and_locations, download_nltk_data

app = Flask(__name__)

# Download NLTK data when the application starts
try:
    download_nltk_data()
except Exception as e:
    print(f"Failed to download NLTK data: {str(e)}")

""" 
instead of repeating the string literal "Medical Equipment" multiple times
defining it as a constant improves code and avoids errors when making changes. 
"""
MEDICAL_EQUIPMENT_KEY = "Medical Equipment"

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running!"})

@app.route('/python', methods=['GET'])
def test_python():
    return { "message": "api called" }

@app.route('/docker', methods=['GET'])
def test_docker():
    return { "message": "api called through docker" }

@app.route('/process_query', methods=['POST'])
def process_query():
    print("Received request in Python service:", request.json)
    try:
        data = request.json
        print("Processing data:", data)
        user_query = data.get('query')
        page = data.get('page', 1)
        items_per_page = data.get('items_per_page', 12)

        if not user_query:
            return jsonify({"error": "Query is required"}), 400

        print(f"User Query: {user_query}")
        
        # Preprocess and fetch necessary data
        tokens = preprocess_query(user_query)
        print("Attempting database connection...")
        medical_equipment, sales_area = fetch_medical_equipment_and_locations()
        print("Database connection successful")
        entities = extract_entities(tokens, medical_equipment, sales_area)

        # Handle case where no entities are found
        if not entities[MEDICAL_EQUIPMENT_KEY]:
            return jsonify({
                "error": f"{MEDICAL_EQUIPMENT_KEY} not found",
                "entities": entities
            }), 404

        # Fetch products
        location = entities['Location'][0] if entities['Location'] else None
        condition = entities['Condition'][0].lower() if entities['Condition'] else None
        price_info = entities['Price']

        # Optimize multi-location handling
        if entities["Location Missing"]:
            products_response = fetch_products(
                entities[MEDICAL_EQUIPMENT_KEY][0],
                None,  # Pass None for location to get all locations
                condition,
                price_info,
                page,
                items_per_page
            )
        else:
            products_response = fetch_products(
                entities[MEDICAL_EQUIPMENT_KEY][0],
                location,
                condition,
                price_info,
                page,
                items_per_page
            )

        logging.basicConfig(
            filename="query_logs.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True
        )

        logging.info(f"Query: {user_query}, Entities: {entities}")
        print("Logging executed!")

        # Handle case where no products are found
        if not products_response['products']:
            return jsonify({
                "error": "No products found matching the query",
                "entities": entities
            }), 404

        return jsonify({
            "success": True,
            "query": user_query,
            "entities": entities,
            **products_response
        })

    except Exception as e:
        print(f"Detailed Python error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)