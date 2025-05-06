import sys, os
from newspaper import Article
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import FeedbackForm, TextAnalysisForm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.comparare import search_trusted_news, compare_with_trusted_news, notify_user
from utils.frecventa_cuvinte import analiza_text, procentaje_cuvinte, comparare_cuvinte, construieste_dict_comune

API_KEY = "d8252bbbbe28439abf4d9739288dde30"

# Create your views here.
def index(request):
    return render(request, 'pagina_principala/index.html')

def feedback_view(request):
    if request.method == "POST":
        form = FeedbackForm(request.POST)
        if form.is_valid():
            form.save()
            form.send_email()
            return redirect('feedback-success')
    else:
        form = FeedbackForm()
    context = { 'form': form }
    return render(request, 'pagina_principala/feedback.html', context)

def feedback_success_view(request):
    return render(request, 'pagina_principala/feedback_success.html')

def analysis_view(request):
    form = TextAnalysisForm(request.POST or None, initial={'input_type': 'text'})
    
    if request.method == "POST" and form.is_valid():
        input_type = form.cleaned_data['input_type']
        if input_type == 'text':
            text_content = form.cleaned_data['text_content']
            # Process the text content
        elif input_type == 'url':
            url_content = form.cleaned_data['url_content']
            # Process the URL content
        
        # Store data in the session
        request.session['input_type'] = input_type
        request.session['content'] = url_content if input_type == 'url' else text_content
        return redirect('result')
    else:
        form = TextAnalysisForm()

    return render(request, 'pagina_principala/analyze.html', {'form': form})

def result_view(request):
    input_type = request.session.get('input_type')
    content = request.session.get('content')
    
    if not input_type or not content:
        return redirect('analysis')  # Redirect back if session data is missing
    
    if input_type == 'url':
        print("Se descarca continutul articolului din URL...")
        article = Article(content)
        article.download()
        article.parse()
        text = article.text
    else:
        text = content
    
    # comparare frecvente
    '''
    propozitii = []
    text = article.text
    dict_cuvinte = analiza_text(text, propozitii)
    procentaje_cuvinte(dict_cuvinte)
    dict_comune = construieste_dict_comune()
    #comparare_cuvinte(dict_cuvinte, dict_comune)
    '''
    
    query = text[:100]
    print(query)

    print("Searching for trusted news...")
    trusted_articles = search_trusted_news(API_KEY, "President Donald TRump")

    if trusted_articles:
        print("Comparing with trusted news...")
        similarity = compare_with_trusted_news(content, trusted_articles)
        notify_user(similarity)
    else:
        print("No trusted articles found.")
    
    # Process the input_type and content
    print(f"Se proceseaza {input_type}: {content[:100]}")
    
    return render(request, 'pagina_principala/result.html', {'input_type': input_type, 'content': content, 'text': text})