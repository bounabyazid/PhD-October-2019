#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:40:49 2019

@author: polo
"""
import json
import requests

def getImgComments(gallery_id):
    url = 'https://api.imgur.com/3/gallery/'+gallery_id+'/comments'

    payload = {'access_token': '984ad7cedbd3a20e7c20ac4b0fecbb138813b73ed'}
    files = {}
    headers = {'Authorization': 'Client-ID 81f0d8191aeec3b'}

    r = requests.request('GET', url, headers = headers, data = payload, files = files)
    j = json.loads(r.text)
    return j['data']

def getGalleryInfo(pictureId):
    #https://code.i-harness.com/fr/q/d59bc4
    header= {"Content-Type": "text", "Authorization": "Client-ID 81f0d8191aeec3b"}
    r = requests.get("https://api.imgur.com/3/gallery/"+pictureId, headers=header)
    j = json.loads(r.text)
    return j['data']

def AllThing(gallery_id):
    Gkeys = ['id','link','description','datetime','downs','ups','favorite_count','section','tags','title','topic','images']
    Data = getGalleryInfo(gallery_id)
    Ginf = {x: Data[x] for x in Gkeys if x in Data}
    
    if 'images' not in Ginf.keys():
        Ginf['Comments'] = getImgComments(gallery_id)
    else:
        Ginf['Comments'] = getImgComments(gallery_id)

    return Ginf

Data = AllThing('GFaDI')
with open('GFaDI.json', 'w') as outfile:
     json.dump(Data, outfile)