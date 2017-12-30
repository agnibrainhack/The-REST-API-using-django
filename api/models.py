# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models


# Create your models here.
class Note(models.Model):
    product_title = models.CharField(max_length=200)
    product_discount = models.TextField()
    product_store = models.TextField(default='Flipkart')

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.product_discount
