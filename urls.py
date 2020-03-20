# encoding: utf-8

from utils.urlparse import include, url_wrapper

urls_patterns = url_wrapper([
    (r"/soccor", include('apps.restful.urls')),
])