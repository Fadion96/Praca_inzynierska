from django.db import models
from django.contrib.postgres.fields import JSONField


# Create your models here.

class ProcessingFunction(models.Model):
    name = models.CharField(max_length=30)
    function = models.CharField(max_length=30)
    number_of_images = models.PositiveSmallIntegerField(default=1)
    params = JSONField(blank=True, null=True)

    def __str__(self):
        return self.name

