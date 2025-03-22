from flask import Flask, render_template, request, jsonify
from model import get_recommendations, analyze_sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    """API Endpoint: Get movie recommendations."""
    data = request.json
    movie_name = data.get("movie")
    if not movie_name:
        return jsonify({"error": "Please provide a movie name!"}), 400
    
    recommendations = get_recommendations(movie_name)
    return jsonify({"recommended_movies": recommendations})

@app.route('/sentiment', methods=['POST'])
def sentiment():
    """API Endpoint: Analyze review sentiment."""
    data = request.json
    review = data.get("review")
    if not review:
        return jsonify({"error": "Please provide a review!"}), 400
    
    sentiment_result = analyze_sentiment(review)
    return jsonify({"sentiment": sentiment_result})

if __name__ == '__main__':
    app.run(debug=True)
