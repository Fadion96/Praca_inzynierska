# Generated by Django 2.1.4 on 2018-12-13 15:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processing', '0003_processingfunction_number_of_images'),
    ]

    operations = [
        migrations.AlterField(
            model_name='processingfunction',
            name='number_of_images',
            field=models.IntegerField(blank=True),
        ),
    ]
