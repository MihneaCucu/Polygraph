from django import forms
from .models import Feedback
from django.core.validators import URLValidator

class FeedbackForm(forms.ModelForm):
    nume = forms.CharField(max_length=100)
    email = forms.EmailField()
    feedback_user = forms.CharField(widget=forms.Textarea)
    
    class Meta:
        model = Feedback
        fields = ['nume', 'email', 'feedback_user']
    
    def send_email(self):
        print(f"Sending email from {self.cleaned_data['email']} with" + 
        f" feedback: {self.cleaned_data['feedback_user']}")

class TextAnalysisForm(forms.Form):
    # Hidden field to track input type
    input_type = forms.CharField(widget=forms.HiddenInput())

    # Separate fields for text and URL inputs
    text_content = forms.CharField(
        label="Text Content",
        required=False,  # Not required because only one field will be used
        widget=forms.Textarea(attrs={'rows': 15, 'cols': 75})
    )
    url_content = forms.URLField(
        label="URL Content",
        required=False,  # Not required because only one field will be used
        widget=forms.URLInput(attrs={'placeholder': 'https://example.com'})
    )

    def clean(self):
        cleaned_data = super().clean()
        input_type = cleaned_data.get('input_type')
        text_content = cleaned_data.get('text_content')
        url_content = cleaned_data.get('url_content')

        if input_type == 'text':
            if not text_content or len(text_content) < 10:
                self.add_error('text_content', 'Text must be at least 10 characters.')
        elif input_type == 'url':
            if not url_content:
                self.add_error('url_content', 'Please enter a valid URL.')
        else:
            self.add_error(None, 'Invalid input type selected.')

        return cleaned_data

