from __future__ import unicode_literals

from django.http import HttpResponse, JsonResponse
from tastypie.resources	 import ModelResource
from api.models import Note
from tastypie.authorization import Authorization

from django.db.models import CharField
from django.db.models.functions import Cast


class NoteResource(ModelResource):
    class Meta:
        queryset = Note.objects.all()
        always_return_data = True
        authorization = Authorization()

    def dehydrate(self, bundle):

        if bundle.request.method == 'POST':
            value = Note.objects.last()
            val = str(value)
            bundle.data['image_result'] = val   #obj.func(val)

        return bundle
