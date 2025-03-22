# Movie Recommendation System

A web-based application that provides movie recommendations and sentiment analysis of reviews using Machine Learning.

## Features
- **Movie Recommendations:** Get recommendations based on a given movie.
- **Sentiment Analysis:** Analyze user reviews to determine sentiment.
- **Interactive UI:** Simple and easy-to-use interface.

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (Python)
- **Machine Learning:** Collaborative & Content-Based Filtering, Sentiment Analysis
- **Database:** MovieLens Dataset

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shailly-450/Movie-Recommendation.git
   cd Movie-Recommendation
   ```
2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python app.py
   ```
5. **Open in browser:**
   - Visit `http://127.0.0.1:5000/`

## API Endpoints
- `POST /recommend` – Returns movie recommendations based on input.
- `POST /sentiment` – Analyzes review sentiment.

## Screenshots
<img width="732" alt="Screenshot 2025-03-22 at 10 14 13 AM" src="https://github.com/user-attachments/assets/9b3b379a-a2c6-48aa-98b1-e3f101d2e73c" />


## Contributing
Feel free to fork this repository, create a feature branch, and submit a pull request!

## License
This project is licensed under the MIT License.

---
🚀 **Enjoy your movie recommendations!**
