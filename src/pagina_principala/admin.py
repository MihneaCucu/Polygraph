from django.contrib import admin
from .models import IncredereSursa, RezultatAnaliza, Feedback, GlobalThreshold
from .models import AllsidesData, UnigramFreq, FormSubmission

class IncredereSursaAdmin(admin.ModelAdmin):
    list_display = ('nume', 'domain', 'scor_incredere', 'bias', 'last_update')
    search_fields = ('nume', 'domain')
    list_filter = ('bias',)

class RezultatAnalizaAdmin(admin.ModelAdmin):
    list_display = ('input_text', 'creat_la', 'sursa')
    search_fields = ('input_text',)
    list_filter = ('creat_la',)

class AllsidesDataAdmin(admin.ModelAdmin):
    list_display = ('news_source', 'rating', 'rating_num', 'type', 'url')
    list_per_page = 20

class UnigramFreqAdmin(admin.ModelAdmin):
    list_display = ('word', 'count')
    list_per_page = 100

class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('analiza', 'email', 'creat_la')
    search_fields = ('feedback_user',)
    list_filter = ('creat_la',)

class FormSubmissionAdmin(admin.ModelAdmin):
    list_display = ('form_data', 'created_at')

class GlobalThresholdAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        # Prevent creating new instances (only 1 allowed)
        return False

    def has_delete_permission(self, request, obj=None):
        # Prevent deletion
        return False


# Register your models here
admin.site.register(IncredereSursa, IncredereSursaAdmin)
admin.site.register(RezultatAnaliza, RezultatAnalizaAdmin)
admin.site.register(Feedback, FeedbackAdmin)
admin.site.register(GlobalThreshold, GlobalThresholdAdmin)
admin.site.register(AllsidesData, AllsidesDataAdmin)
admin.site.register(UnigramFreq, UnigramFreqAdmin)
admin.site.register(FormSubmission, FormSubmissionAdmin)
