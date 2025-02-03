# Step 1: Use a Python base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Step 4: Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a script to download NLTK data
RUN echo 'import nltk; \
    nltk.download("punkt", download_dir="/usr/share/nltk_data"); \
    nltk.download("stopwords", download_dir="/usr/share/nltk_data"); \
    nltk.download("wordnet", download_dir="/usr/share/nltk_data"); \
    nltk.download("omw-1.4", download_dir="/usr/share/nltk_data")' > download_nltk.py

# Run the download script
RUN python download_nltk.py

# Step 3: Copy the Python files into the container
COPY . .

# Create NLTK data directory
RUN mkdir -p /usr/share/nltk_data

# Set environment variable for NLTK data path
ENV NLTK_DATA /usr/share/nltk_data

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"


# Step 5: Expose the port the app will run on
EXPOSE 5000

# Step 6: Set the command to run the app
CMD ["python", "nlp_service.py"]
