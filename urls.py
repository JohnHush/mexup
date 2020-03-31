# encoding: utf-8

from utils.urlparse import include, url_wrapper

urls_patterns = url_wrapper([
    (r"/soccer", include('apps.restful.urls')),
])