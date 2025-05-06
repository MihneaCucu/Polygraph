from django.contrib import admin
from .models import IncredereSursa, RezultatAnaliza, Feedback

# Customize IncredereSursa admin
class IncredereSursaAdmin(admin.ModelAdmin):
    list_display = ('nume', 'domain', 'scor_incredere', 'bias', 'last_update')
    search_fields = ('nume', 'domain')
    list_filter = ('bias',)

# Customize RezultatAnaliza admin
class RezultatAnalizaAdmin(admin.ModelAdmin):
    list_display = ('input_text', 'creat_la', 'sursa')
    search_fields = ('input_text',)
    list_filter = ('creat_la',)

# Customize Feedback admin
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('analiza', 'email', 'creat_la')
    search_fields = ('feedback_user',)
    list_filter = ('creat_la',)

# Register your models here
admin.site.register(IncredereSursa, IncredereSursaAdmin)
admin.site.register(RezultatAnaliza, RezultatAnalizaAdmin)
admin.site.register(Feedback, FeedbackAdmin)
