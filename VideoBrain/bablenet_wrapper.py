# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:36:26 2020

@author: vegex
"""

import urllib
import json
import gzip
import emoji

from io import BytesIO


class BabelNetWrapper(object):
    
    
    def __init__(self, key, lang = 'EN'):
        self.key = key
        self.lang = lang
        
        
    
    def __getIDs(self, lemma):
        service_url = 'https://babelnet.io/v5/getSynsetIds'
        params = {'lemma' : lemma,
                  'searchLang' : self.lang,
                  'key'  : self.key
                  }
        url = service_url + '?' + urllib.parse.urlencode(params)
        request = urllib.request.Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urllib.request.urlopen(request)
        IDs = list()
        if response.info().get('Content-Encoding') == 'gzip':
            buf = BytesIO(response.read())
            f = gzip.GzipFile(fileobj=buf)
            data = json.loads(f.read())
            for result in data:
                IDs.append(result['id'])
        return IDs



    def getBabelSense(self, lemma, emojies = False):
        service_url = 'https://babelnet.io/v5/getSenses'
        params = {'lemma' : lemma,
                  'searchLang' : self.lang,
                  'key'  : self.key
                  }
        url = service_url + '?' + urllib.parse.urlencode(params)
        request = urllib.request.Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urllib.request.urlopen(request)
        if response.info().get('Content-Encoding') == 'gzip':
            buf = BytesIO(response.read())
            f = gzip.GzipFile(fileobj=buf)
            data = json.loads(f.read())
            senses = set()
            for result in data:
                print(result, '\n\n')
                lemma = result.get('properties').get('simpleLemma')
                if (lemma in emoji.UNICODE_EMOJI) and not emojies:
                    continue
                senses.add(lemma.lower())
        return list(senses)



key = 'f2d097b0-ed98-4ec9-b780-995b95823829'
wrapper = BabelNetWrapper(key)
senses = wrapper.getBabelSense('orange')

    


