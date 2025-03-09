import datetime
import psycopg2
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
import ssl

#* Load environment variables
# load_dotenv()

# Check the environment type
env_type = os.getenv('ENV_TYPE', 'local')  # default to 'local' if not set

# Load the appropriate .env file
if env_type == 'docker':
    load_dotenv('.env')  # for Docker environment
else:
    load_dotenv('.env.local')  # for local development environment

# Check if running in production (Render)
is_production = env_type == "render"

#* Connect to PostgreSQL database
# def get_db_connection():
#     return psycopg2.connect(
#         host=os.getenv('DB_HOST'),
#         database=os.getenv('DB_NAME'),
#         user=os.getenv('DB_USER'),
#         password=os.getenv('DB_PASSWORD'),
#         port=os.getenv('DB_PORT')
#         # for local machine use host="localhost" then for docker as below,
#         # host="postgres",
#         # database="supplier_dashboard",
#         # user="postgres",
#         # password="zxcvbnmq",
#         # port=5432
#     )

def get_db_connection():
    try:
        if is_production:
            # Use Render's DATABASE_URL
            connection_string = os.getenv("DATABASE_URL")
            conn = psycopg2.connect(connection_string, sslmode="require")
        else:
            # Use local environment variables
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                port=os.getenv("DB_PORT")
            )

        print("✅ Database connection successful")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def download_nltk_data():
    try:
        # Handle SSL issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Ensure the NLTK data directory is configured correctly
        nltk_data_path = os.getenv("NLTK_DATA", "/usr/share/nltk_data")
        if not os.path.exists(nltk_data_path):
            os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)

        # Download NLTK resources
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")
        raise


#* Fetch distinct medical equipment and sales areas from the database
def fetch_medical_equipment_and_locations():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT DISTINCT name FROM productsnew")
    medical_equipment = [row[0] for row in cur.fetchall()]

    cur.execute("SELECT DISTINCT sales_area FROM productsnew")
    sales_area = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()
    
    return medical_equipment, sales_area

#* Initialize NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

def init_nltk_resources():
    # This will be called after download_nltk_data()
    global stop_words, lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

# Place these at module level
stop_words = None
lemmatizer = None

# Call download_nltk_data and initialize resources
download_nltk_data()
init_nltk_resources()

#* Preprocess query for tokenization and lemmatization
# def preprocess_query(query):
#     # Convert to lowercase and tokenize
#     tokens = word_tokenize(query.lower())
#     # Remove stopwords but keep price-related words
#     price_words = {'below', 'above', 'under', 'over', 'between', 'from', 'to'}
#     filtered_tokens = [word for word in tokens if word.lower() not in stop_words or word.lower() in price_words]
#     return filtered_tokens

def preprocess_query(query):
    try:
        # Tokenize the query
        tokens = word_tokenize(query.lower())
    except LookupError as e:
        print(f"Tokenizer error: {str(e)}")
        raise RuntimeError("NLTK 'punkt' tokenizer not found. Ensure it is downloaded.")

    # Define stopwords and price-related words
    price_words = {'below', 'above', 'under', 'over', 'between', 'from', 'to'}
    filtered_tokens = [
        word for word in tokens if word not in stop_words or word in price_words
    ]

    # Optionally lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return lemmatized_tokens

#* v2 Preprocess entities for singular/plural reduction
def preprocess_entity(entity_list):
    entity_tokens = {}
    for entity in entity_list:
        tokens = word_tokenize(entity.lower())
        entity_tokens[entity] = set(tokens)
    return entity_tokens


#* Match tokens with entities
def exact_match_entity(tokens, entity_list):
    entity_tokens = preprocess_entity(entity_list)
    matched_entities = set()
    token_set = set(token.lower() for token in tokens)
    
    for entity, entity_set in entity_tokens.items():
        if entity_set.issubset(token_set):
            matched_entities.add(entity)
    
    return list(matched_entities)


#* Analyze tokens sequentially to determine price operation and values
def analyze_price_tokens(tokens):
    price_operators = {
        'below': 'under',
        'under': 'under',
        'less': 'under',
        'above': 'above',
        'over': 'above',
        'greater': 'above',
        'between': 'between',
        'from': 'between'
    }
    
    numbers = []
    operator = None
    i = 0
    
    while i < len(tokens):
        token = tokens[i].lower()
        
        # Check for price operator
        if token in price_operators:
            operator = price_operators[token]
        
        # Extract number value
        # Remove $ and comma from token if present
        cleaned_token = token.replace('$', '').replace(',', '')
        try:
            number = float(cleaned_token)
            numbers.append(number)
        except ValueError:
            pass
        
        i += 1
    
    # If we found numbers but no operator, default to 'under'
    if numbers and not operator:
        operator = 'under'
    
    return operator, numbers


#* Extract price information using sequential analysis
def extract_price_info(tokens):
    operator, numbers = analyze_price_tokens(tokens)
    
    if not numbers:
        return {
            "from": None,
            "to": None,
            "type": None
        }
    
    # Handle different operator cases
    if operator == 'between' and len(numbers) >= 2:
        return {
            "from": min(numbers[0], numbers[1]),
            "to": max(numbers[0], numbers[1]),
            "type": "between"
        }
    elif operator == 'above':
        return {
            "from": numbers[0],
            "to": None,
            "type": "above"
        }
    elif operator == 'under':
        return {
            "from": None,
            "to": numbers[0],
            "type": "under"
        }
    
    # Default case
    return {
        "from": None,
        "to": numbers[0],
        "type": "under"
    }


#* Extract entities from query tokens
def extract_entities(tokens, medical_equipment, sales_area):
    medical_matches = exact_match_entity(tokens, medical_equipment)
    location_matches = exact_match_entity(tokens, sales_area)
    condition_matches = [word for word in tokens if word.lower() in ["used", "new"]]
    price_info = extract_price_info(tokens)

    location_missing = len(location_matches) == 0  # Check if no location is found

    price_related = {'$', 'price', 'from', 'to', 'between', 'and', 'over', 'above', 'under', 'below'}
    matched_tokens = set()
    for match in medical_matches + location_matches + condition_matches:
        matched_tokens.update(word.lower() for word in word_tokenize(match))
    
    unmatched_tokens = [
        token for token in tokens 
        if token.lower() not in matched_tokens 
        and token.lower() not in price_related
        and not any(c.isdigit() for c in token)
    ]
    
    response = {
        "Medical Equipment": medical_matches,
        "Location": location_matches,
        "Condition": condition_matches,
        "Price": price_info,
        "Other": unmatched_tokens,
        "Location Missing": location_missing  # Add this flag
    }
    
    return response

#* Fetch products based on extracted entities
def fetch_products(medical_equipment, location=None, condition=None, price_info=None, page=1, items_per_page=12):
    conn = get_db_connection()
    cur = conn.cursor()

    # Base query to count total results
    count_query = "SELECT COUNT(*) FROM productsnew WHERE name ILIKE %s"
    count_params = [f"%{medical_equipment}%"]

    # Main query with pagination
    query = "SELECT * FROM productsnew WHERE name ILIKE %s"
    params = [f"%{medical_equipment}%"]

    # Apply additional filters
    if location is not None:
        query += " AND sales_area ILIKE %s"
        count_query += " AND sales_area ILIKE %s"
        params.append(f"%{location}%")
        count_params.append(f"%{location}%")

    if condition:
        query += " AND condition ILIKE %s"
        count_query += " AND condition ILIKE %s"
        params.append(f"%{condition}%")
        count_params.append(f"%{condition}%")

    if price_info and price_info.get("type"):
        if price_info["type"] == "between":
            query += " AND CAST(price AS DECIMAL) BETWEEN %s AND %s"
            count_query += " AND CAST(price AS DECIMAL) BETWEEN %s AND %s"
            params.extend([price_info["from"], price_info["to"]])
            count_params.extend([price_info["from"], price_info["to"]])
        elif price_info["type"] == "under":
            query += " AND CAST(price AS DECIMAL) <= %s"
            count_query += " AND CAST(price AS DECIMAL) <= %s"
            params.append(price_info["to"])
            count_params.append(price_info["to"])
        elif price_info["type"] == "above":
            query += " AND CAST(price AS DECIMAL) >= %s"
            count_query += " AND CAST(price AS DECIMAL) >= %s"
            params.append(price_info["from"])
            count_params.append(price_info["from"])

    # Calculate total results and pages
    cur.execute(count_query, count_params)
    total_results = cur.fetchone()[0]
    total_pages = math.ceil(total_results / items_per_page)

    # Add pagination to main query
    offset = (page - 1) * items_per_page
    query += " LIMIT %s OFFSET %s"
    params.extend([items_per_page, offset])

    print("Executing Query:", query)
    print("With Parameters:", params)
    
    cur.execute(query, params)
    results = cur.fetchall()

    cur.close()
    conn.close()

    formatted_results = []
    for row in results:
        formatted_product = {
            'Product_ID': row[0],
            'Name': row[1],
            'Brand': row[2],
            'Model': row[3],
            'CategoryId': row[4],
            'SubcategoryId': row[5],
            'Price': str(row[9]),
            'Condition': row[7],
            'Year': row[8],
            'Location': row[10],
            'Created_At': row[12].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row[12], datetime.datetime) else row[12],
            'Image_URL': row[14],
        }
        formatted_results.append(formatted_product)

    return {
        'products': formatted_results,
        'total_results': total_results,
        'total_pages': total_pages,
        'current_page': page,
        'items_per_page': items_per_page
    }
