# Generated by Django 5.2.3 on 2025-06-13 20:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pagina_principala', '0006_alter_feedback_analiza'),
    ]

    operations = [
        migrations.CreateModel(
            name='GlobalConfig',
            fields=[
                ('id', models.AutoField(editable=False, primary_key=True, serialize=False)),
                ('threshold', models.FloatField(default=0.5, help_text='Algorithm threshold for satisfiable answers')),
            ],
            options={
                'verbose_name': 'Global Configuration',
            },
        ),
    ]
