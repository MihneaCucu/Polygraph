from django.db import models
from django.db.models import JSONField

# Create your models here.

class IncredereSursa(models.Model):
    """"Contine surse de incredere"""
    nume = models.CharField(max_length=200)
    domain = models.CharField(max_length=100, unique=True)
    scor_incredere = models.DecimalField(max_digits=3, decimal_places=2, default=0.5)
    bias = models.CharField(max_length=20, choices=[
        ('stanga', 'Inclinari de stanga'),
        ('dreapta', 'Inclinari de dreapta'),
        ('fdreapta', 'Inclinari puternice dreapta'),
        ('fstanga', 'Inclinari puternice stanga'), 
        ('centru', 'Neutru')
    ])
    last_update = models.DateTimeField(auto_now=True)


class RezultatAnaliza(models.Model):
    """Contine rezultatul analizei linkului(posibil si text??)"""
    input_text = models.TextField()
    creat_la = models.DateTimeField(auto_now_add=True)
    
    sentiment = JSONField(null=True, blank=True)  # Stores {'label': 'positive', 'score': 0.95}
    veridicitate = JSONField(null=True, blank=True)  # Stores {'label': 'fake', 'score': 0.87}
    verificare_sursa = JSONField(null=True, blank=True)  # Stores {'is_trusted': False, 'reason': 'Unknown source'}
    articole_similare = JSONField(null=True, blank=True)  # Stores list of found articles
    
    sursa = models.ForeignKey(
        IncredereSursa,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )


class Feedback(models.Model):
    """User feedback"""
    analiza = models.ForeignKey(RezultatAnaliza, on_delete=models.CASCADE, null=True, blank=True)
    nume = models.CharField(max_length=200)
    email = models.CharField(max_length=150)
    feedback_user = models.TextField(blank=True)
    creat_la = models.DateTimeField(auto_now_add=True)