from django.shortcuts import render
from django.views import View
from .models import Media

class Captioner(View):
    def get(self, request, *args, **kwargs):
        data = {
            'link': 'https://picsum.photos/800',
            'caption': 'Amazing Caption!!'
        }
        return render(request,'index.html',data)

    def post(self, request, *args, **kwargs):
        img = request.FILES['img_logo']
        m = Media(caption="Cat with moustache",media=img)
        m.save()
        return render(request,'index.html')

class Home(View):
    def get(self,request,*args,**kwargs):
        media_list = Media.objects.all()
        data = {
            'media_list': media_list,
        }
        return render(request,'list.html', context=data)
