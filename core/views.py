from django.shortcuts import render, redirect
from django.views import View
from .models import Media
# from .caption import analyser
class Captioner(View):
    def get(self, request, *args, **kwargs):
        data = {
            'link': 'https://picsum.photos/800',
            'caption': 'Amazing Caption!!'
        }
        return render(request,'caption.html',data)

    def post(self, request, *args, **kwargs):
        img = request.FILES['img_logo']
        m = Media(media=img)
        try:
            m.save()
        except:
            return redirect('error')
        return redirect('browse')

class Home(View):
    def get(self,request,*args,**kwargs):
        media_list = Media.objects.all().order_by('-id')
        data = {
            'media_list': media_list,
        }
        return render(request,'list.html', context=data)
    
    def post(self, request, *args, **kwargs):
        query = request.POST['query']
        media_list = Media.objects.filter(caption__icontains=query).order_by('-id')
        data = {
            'media_list': media_list,
        }
        return render(request,'list.html', context=data)

class Error(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'error.html')
        
