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


class GlobalThreshold(models.Model):
    # Singleton manager pentru crearea unei singure instante
    class Meta:
        verbose_name = "Global Threshold"

    id = models.AutoField(primary_key=True, editable=False)
    threshold = models.FloatField(
        default=0.5,  # Valoare default
        help_text="Threshold pentru a defini procentul la care o sursa este considerata FAKE/REALA"
    )

    def save(self, *args, **kwargs):
        # Fortam o singura instanta
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls):
        # Primeste sau creeaza instanta
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class AllsidesData(models.Model):
    id = models.AutoField(primary_key=True)
    
    news_source = models.CharField(max_length=255)
    rating = models.CharField(max_length=255)
    rating_num = models.CharField(max_length=255)
    type = models.CharField(max_length=255)
    agree = models.CharField(max_length=255)
    disagree = models.CharField(max_length=255)
    perc_agree = models.CharField(max_length=255)
    url = models.CharField(max_length=255)
    editorial_review = models.CharField(max_length=255)
    blind_survey = models.CharField(max_length=255)
    third_party_analysis = models.CharField(max_length=255)
    independent_research = models.CharField(max_length=255)
    confidence_level = models.CharField(max_length=255)
    twitter = models.CharField(max_length=255)
    wiki = models.CharField(max_length=255)
    facebook = models.CharField(max_length=255)
    screen_name = models.CharField(max_length=255)

    class Meta:
        managed = False  
        db_table = 'pagina_principala_allsides_data'  


class UnigramFreq(models.Model):
    word = models.CharField(max_length=255)
    count = models.CharField(max_length=255)

    class Meta:
        managed = False  
        db_table = 'pagina_principala_unigram_freq'