import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def normalize_text(text):
    text = re.sub(r'\W+', ' ', text)
    return text.strip().lower()

def search_trusted_news(api_key, query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        if not articles:
            print("No articles found for the given query.")
        #print(articles[0]["content"])
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
        return (f"The news is consistent with trusted sources (similarity: {similarity:2f}).")
    else:
        return (f"Significant differences found compared to trusted sources (similarity: {similarity:2f}).")
    
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
    
    output = ""
    if suggestions:
        output += "\nSuggested alternative trusted sources:\n"
        for source, title, url in suggestions:
            output += f"- {source}: {title}\n  {url}\n"
    else:
        output = "No alternative sources found."
    
    return suggestions

if __name__ == "__main__":
    api_key = "d8252bbbbe28439abf4d9739288dde30"
    input_text = "The Earth revolves around the Sun in an elliptical orbit, completing one revolution approximately every 365.25 days. This motion, known as Earth's revolution, is responsible for the changing seasons. The phenomenon has been confirmed by centuries of astronomical observations and scientific research."
    query = "Earth Sun orbit"

    print("Searching for trusted news...")
    trusted_articles = search_trusted_news(api_key, query)

    if trusted_articles:
        print("Comparing with trusted news...")
        similarity = compare_with_trusted_news(input_text, trusted_articles)
        notify_user(similarity)
    else:
        print("No trusted articles found.")