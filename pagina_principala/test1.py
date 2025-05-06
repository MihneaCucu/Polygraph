from newspaper import Article
article = Article('https://edition.cnn.com/2025/05/05/business/trade-war-deal-trump')
article.download()
article.parse()
print(article.text)