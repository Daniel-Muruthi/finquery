from django.shortcuts import render
from django.http import JsonResponse
from .utils import categorize_intent
from .linear_svc_utils import categorize_intent as linear_categorize_intent

def categorizer_home(request):
    """
    Render the BERT categorizer page
    """
    return render(request, 'categorizer/bert_home.html', {'model_name': 'BERT'})

def linear_svc_home(request):
    """
    Render the LinearSVC categorizer page
    """
    return render(request, 'categorizer/linear_svc_home.html', {'model_name': 'LinearSVC'})

def api_categorize(request):
    """
    API endpoint to categorize query text using BERT model
    """
    if request.method == 'POST':
        query_text = request.POST.get('query', '')
        if not query_text:
            return JsonResponse({'error': 'No query provided'}, status=400)
        
        try:
            # Call the categorize_intent function from utils.py (BERT)
            category = categorize_intent(query_text)
            return JsonResponse({'category': category})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

def api_linear_svc_categorize(request):
    """
    API endpoint to categorize query text using LinearSVC model
    """
    if request.method == 'POST':
        query_text = request.POST.get('query', '')
        if not query_text:
            return JsonResponse({'error': 'No query provided'}, status=400)
        
        try:
            # Call the categorize_intent function from linear_svc_utils.py
            category = linear_categorize_intent(query_text)
            return JsonResponse({'category': category})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST requests allowed'}, status=405)