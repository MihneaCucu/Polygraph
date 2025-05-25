import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import argparse

def normalize_text(text):
    text = re.sub(r'\W+', ' ', text)
    return text.strip().lower()

def extract_keywords(text, num_keywords=5):
    words = [w for w in re.findall(r'\b\w+\b', text.lower()) if w not in ENGLISH_STOP_WORDS]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq, key=freq.get, reverse=True)
    return ' '.join(sorted_words[:num_keywords])

def search_trusted_news(api_key, query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        if not articles:
            print("No articles found for the given query.")
        return [article["content"] for article in articles if article.get("content")], articles
    else:
        print(f"Error fetching news: {response.status_code}, Response: {response.text}")
        return [], []

def compare_with_trusted_news(input_text, trusted_articles):
    vectorizer = TfidfVectorizer().fit_transform([input_text] + trusted_articles)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    max_similarity = max(similarities) if similarities.size > 0 else 0
    return max_similarity

def notify_user(similarity, threshold=0.7):
    if similarity >= threshold:
        print(f"The news is consistent with trusted sources (similarity: {similarity:.2f}).")
    else:
        print(f"Significant differences found compared to trusted sources (similarity: {similarity:.2f}).")

def suggest_alternative_sources(articles, max_sources=3):
    # Prioritize diverse sources
    seen_sources = set()
    suggestions = []
    for article in articles:
        source = article.get("source", {}).get("name", "Unknown")
        url = article.get("url", None)
        title = article.get("title", "No title")
        if source not in seen_sources and url:
            suggestions.append((source, title, url))
            seen_sources.add(source)
        if len(suggestions) >= max_sources:
            break
    if suggestions:
        print("\nSugested alternative trusted sources:")
        for source, title, url in suggestions:
            print(f"- {source}: {title}\n  {url}")
    else:
        print("No alternative sources found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold for credibility')
    parser.add_argument('--max_sources', type=int, default=3, help='Number of alternative sources to suggest')
    parser.add_argument('--query', type=str, default="Earth Sun orbit", help='Query for trusted news')
    parser.add_argument('--input_file', type=str, default=None, help='Path to input text file')
    args = parser.parse_args()

    api_key = "d8252bbbbe28439abf4d9739288dde30"
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        input_text = "The Earth revolves around the Sun in an elliptical orbit, completing one revolution approximately every 365.25 days. This motion, known as Earth's revolution, is responsible for the changing seasons. The phenomenon has been confirmed by centuries of astronomical observations and scientific research."

    if args.query == "Earth Sun orbit" or not args.query:
        query = extract_keywords(input_text)
        print(f"Generated query: {query}")
    else:
        query = args.query

    print("Searching for trusted news...")
    trusted_articles, articles = search_trusted_news(api_key, query)

    if trusted_articles:
        print("Comparing with trusted news...")
        similarity = compare_with_trusted_news(input_text, trusted_articles)
        notify_user(similarity, threshold=args.threshold)
        suggest_alternative_sources(articles, max_sources=args.max_sources)
    else:
        print("No trusted articles found.")

