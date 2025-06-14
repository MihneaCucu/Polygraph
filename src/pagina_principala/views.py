import sys, os
import joblib
import json
from pathlib import Path
from newspaper import Article
from django.core.paginator import Paginator
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import FeedbackForm, TextAnalysisForm
from .models import GlobalThreshold, AllsidesData

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.comparare import search_trusted_news, compare_with_trusted_news, notify_user, suggest_alternative_sources
from utils.frecventa_cuvinte import analiza_text, procentaje_cuvinte, comparare_cuvinte, construieste_dict_comune
from utils.clasificare import predict_from_file_with_nlp
from utils.identificator_bias import review
from utils.rezumat import rezumat_text

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
        elif input_type == 'url':
            url_content = form.cleaned_data['url_content']
        
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
    config = GlobalThreshold.load()
    
    if not input_type or not content:
        return redirect('analysis')  # Redirect back if session data is missing
        
    if input_type == 'url':
        # Incredere sursa din baza de date
        incredere_sursa = review(content)
        if incredere_sursa is None:
            incredere_sursa = "Baza noastră de date nu a găsit această sursă"
        # Descarcare continut text articol
        print("Se descarca continutul articolului din URL...")
        article = Article(content)
        article.download()
        article.parse()
        text = article.text
    else:
        text = content
        incredere_sursa = "Ați introdus un text anonim și nu un link. Sursă necunoscută"
    
    # Frecventa cuvinte/ cuvinte cheie   
    propozitii = []
    dict_cuvinte = analiza_text(text, propozitii)
    procentaje_cuvinte(dict_cuvinte)
    dict_comune = construieste_dict_comune()
    query, frecventa = comparare_cuvinte(dict_cuvinte, dict_comune)
    
    # Comparare cu surse de incredere
    print("Searching for trusted news...")
    trusted_articles, articles = search_trusted_news(API_KEY, query)
    print("ARTICOLe DE INCREDERE [0]\n")
    print(trusted_articles[0])

    if trusted_articles:
        print("Comparing with trusted news...")
        similarity = compare_with_trusted_news(content, trusted_articles)
        comparare_stiri = notify_user(similarity)
        surse_alternative = suggest_alternative_sources(articles, max_sources=3)
    else:
        comparare_stiri = "No trusted articles found."  
    
    # Clasificare in functie de model
    model_path = Path(__file__).resolve().parent.parent / "pagina_principala" / "ai_models" / "model_clasificare_0.joblib"
    model = joblib.load(model_path)
    # Threshold credibilitate
    threshold = config.threshold
    # print(threshold)

    print("\nAnalysing input...")
    predictie_stire = predict_from_file_with_nlp(model, text, threshold)
    #print(predictie_stire)
    
    # Afisare tip de continut folosit
    print(f"S-a folosit {input_type}: {content[:100]}")
    
    return render(
            request, 
            'pagina_principala/result.html', 
            {'input_type': input_type, 'incredere_sursa': incredere_sursa, 'surse_alternative': surse_alternative,
            'text': text, 'frecventa': frecventa, 'comparare_stiri': comparare_stiri,
            'predictie_stire': predictie_stire}
            )


# @csrf_exempt 
def get_summary(request):
    if request.method == 'POST':
        try:
            # Parse JSON
            data = json.loads(request.body.decode('utf-8'))
            text = data.get('text', '')
            
            # print("TEXTUL SUPUS REZUMATULUI:\n")
            # print(text)
            
            summary = rezumat_text(text)
            
            return JsonResponse({
                'summary': summary,
                'status': 'success'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON data',
                'status': 'error'
            }, status=400)
            
        except Exception as e:
            return JsonResponse({
                'error': str(e),
                'status': 'error'
            }, status=500)
    
    return JsonResponse({
        'error': 'Only POST requests are allowed',
        'status': 'error'
    }, status=405)


def entry_list(request):
    entries_list = AllsidesData.objects.all()
    
    # Paginare
    paginator = Paginator(entries_list, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'entries': page_obj,
        'page_obj': page_obj
    }
    return render(request, 'pagina_principala/entries.html', context)

def about_view(request):
    return render('request', 'pagina_principala/about.html')
