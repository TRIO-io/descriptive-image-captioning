from django.db import models

class Media(models.Model):
    caption = models.CharField(max_length=200)
    media = models.FileField()

    def __str__(self):
        return self.caption