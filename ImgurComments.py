#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:42:36 2019
@author: polo
"""

#https://gist.github.com/openfly/1577a31751ccb87bf86f

#client id = 81f0d8191aeec3b
#Client Secret = 84ad7cedbd3a20e7c20ac4b0fecbb138813b73ed

import os
import csv
import time
import json
import requests
import urllib.request
import urllib
import codecs
import errno

import numpy as np
import pandas as pd

#url = 'http://imgur.com/gallery/DVNWyG8'
#payload = {}
#headers = {'Authorization': 'fa4f1a3c3b581943c185247b3a0b700a544e8aad'}


#url = "https://api.imgur.com/3/album/{{albumHash}}"

headers = {'Authorization': 'Client-ID {{81f0d8191aeec3b}}'}

#response = requests.request("GET", url, headers=headers)

#print(response.text)

imgur_client = "81f0d8191aeec3b"
imgur_secret = "984ad7cedbd3a20e7c20ac4b0fecbb138813b73ed"

#imgur_url = "https://api.imgur.com"
#username = "vissago"

def count_comments(username, imgur_client):

    header= {"Content-Type": "text", "Authorization": "Client-ID " + imgur_client}
    r = requests.get("https://api.imgur.com/3/account/" + username +"/comments/count.json", headers=header)
    j = json.loads(r.text)
    count = j['data']
    return count

def acquire_comments(username, imgur_client):
    count = count_comments(username, imgur_client)
    if ( count > 0):
        print ("count of commentary is %s" % count)
        pages = int(count / 50)
        finger = 0
        f = codecs.open(username + ".csv", encoding='utf-8', mode='w+')
        while ( finger < pages ):
            header= {"Content-Type": "text", "Authorization": "Client-ID " + imgur_client}
            r = requests.get("https://api.imgur.com/3/account/" + username + "/comments/newest/" + str(finger) + ".json", headers=header)
            comments = json.loads(r.text)
            for comment in comments['data']:
                #pprint.pprint( comment, indent=4 )
                string = comment['comment']
                datetime = comment['datetime']
                points = comment['points']
                imageid = comment['image_id']
                commentid = comment['id']
                # https://api.imgur.com/3/comment/{id}/vote/{vote}
                r = requests.get("https://api.imgur.com/3/comment/" + str(commentid) + "/vote/up", headers=header)
                print ("\033[32mUPVOTE\033[0m: %s : %s \n" % (commentid, string))
                time.sleep(5)
                outputline = "%s, %s, %s, %s, %s\n" % (username, imageid, datetime, points, string)
                f.write(outputline)

            finger += 1
        f.seek(0)
        print (repr(f.readline()[:1]))
        f.close()
    else:
        print ("%s has no commentary... must be the gruqq =(\n")

#acquire_comments(username, imgur_client)

'http://instagram.com/travel_wheretonext'
'http://travelwheretonext.com'

Countries = [
            'Algeria','Tunisia','Morocco','Namibia','Finland','Mexico','Bolivia','Arizona','California','Egypt',
            'China','Hawaii','Brazil','France','Spain','Italy','Switzerland','India','Australia','Maldives',
            'Dubai','Russia','Thailand','Colombia','Ecuador','Germany','South Africa','Netherlands','Panama','Argentina',
            'Albania','Austria','Cameroon','Chile','Congo','Costa Rica','Croatia','Nigeria','Portugal','Mongolia',
            'Kazakhstan', 'Libya', 'Madagascar', 'Jamaica', 'Indonesia', 'Cambodia','Guatemala','Greece',
            #'', '', '', '', '', '','', '','','',
            #'', '', '', '', '', '','', '','','',
            #'', '', '', '', '', '','', '','','',
            #'', '', '', '', '', '','', '','','',
            #'', '', '', '', '', '','', '','','',
            ]

galeries = [
            #Algeria
            'x6TwpSQ','R3hT2v0','ZTGHc','U6LOSWO','AjChT','0fGGq','ziaXx','joMyrTQ','6aCY1be','DDaTZ1s',
            #Tunisia
            '75rbKac','bvxZ8Th','djCGk6D','ahPtwMS','vk6QXAM','PJv4BvC','skJPplY','HeTx7bI','esZmeew','MTRo5',
            #Morocco
            'k8QL66z','xM2YZ4r','hVQig0o','ka5IiPV','bYiRaqg','9QawVw1','bMklK','4fu8moV','MSpeBeV','3ZiZJ',
            #Namibia
            'E5A3H2p','0ZwNLog','mAJR61Q','V4jmZKf','9nu2K','bO1oiuK','Ltumd','ngRD6Eu','T89estS','c2fSBez',
            #Finland
            'J5hHYue','d7XYT6X','QxcPfDw','LZnJJ9s','7IJXO','8N7vk','R4Ty7','ma1PO4W','du4L6sy','76Ooc',
            #Mexico
            'HjW7KTM','dPCbRXt','ye4Lfow','fMegtI3','zZ2o7','867Sv6G','5586I','NlAQQ8q','rcpULdg','ZQ66AHs',
            #Bolivia
            'PpN8ZYn','mwLdDmt','tOYZI15','XNX4U','FXLkHcD','E2cJkFX','mK6dWzg','fJCxZTa','lY15s','cTj6Dlj',
            #Arizona
            'MrelgvY','spfNPNO','PLn0Owa','QI4U9','i56JLpa','X2Cn4ZU','y5CDWGk','1B7CWyz','lTvZAQb','WJxgG',
            #California
            'NRAjuGF','Lvh407h','BdnpW','HRoAKgZ','m5KaMVP','aiBgw4H','S84Q9QN','JWWfhpp','1S0iQ1v','kmAu4Hv',
            #egypt
            'tB4mi','gonkB','n80hA','efFH5RZ','NWYZans','phuY3i5','J2muY','lOBda27','JaIihJx','bAy3vQR',
            #China 
            'cwLSAoz','5xJiRFp','DW0iIDI','NjBMNxQ','FXErKGJ','9hDeL37','hyO7iLB','Oj2Rb','1yWxOji','4LN5t0q',
            #Hawaii
            'rRKKhZF','syczAV4','ZM2yJzP','0FFKc','Kk9n2hL','TvMWE','VyRCtdr','TFYEj','jh4AH','9UtCG6k',
            #Brazil
            'V4G3HQx','S44YI5R','xsX5pyC','xyfJufu','rWFGSpv','6S9f8o7','ngS3V','COAND','bwmsg7t','nnLYQ8r',
            #France
            'MHrnQCe','PzpX4','RdgMIOZ','FiIvycZ','wHsc3q6','5vfJJiR','ziUAdmQ','6v8HmHE','j3G6T','CLLDz',
            #Spain
            'T22O5TO','9Yw2SNu','uXYqWHO','o4PTnPV','vcwrYpX','xVVgyIg','TvHgfey','e3U8O','Ice3sai','VdyH6',
            #Italy
            'c0AZWIS', 'OadXq', 'bm5d9tr', 'IaM0KwT', 'qkXuV9R', '1WJ2oAA','TXjepX0','2MMPfGr','5um4InX','U2MVdYz',
            #Switzerland
            'vRLD62o','9hOdzsx','0GqOUs6','Q89Sg','LrEK3o8','eRM6hHb','dAKfsJx','4EUZ6mi','eLHPzsR','SlTRt',
            #India
            'J0aS5e2','2lxW8ZK','GKwiiWK','CMulTKf','zj9IjCp','fgR0tGr','Ep4Us','kKtnD','SNHHd','DMou6L9',
            #Australia
            'pBN3g','AoDUIXN','LX9OS','qHxKANc','u1VRFN2','lauGwqr','sN6Y5WH','x5cQT2j','jKLn6UE','lze5Ri3',
            #Maldives
            '2Vn789J','yyEb3X2','aKVjDox','CxvV6A5','s058p77','9qOeass','6xtXgHp','iaYmOgf','4QLi0UI','iiLcAjM',
            #Dubai
            '3DFWMYu','QaUaE6t','WnQyqfo','Zdky8JL','NSrKs','qxChgJm','hdzELGJ','hmpah','CfG7H0F','n8FMpoV',
            #Russia
            '1YjteIr','ZTgIsMx','VT3Hd','gUjLLN3','LTgWPb3','uKnMvyp','ciu0V','Ts5W6to','o6YXsvx','TzomF',
            #Thailand
            '92JMtKi','Es7VHhT','24e466c','gCVE8J5','EkvaUDH','Hh37R1A','Szr9h','o0OauLV','WG1Yg1H','YvjbdNW',
            #Colombia
            '7Q87LqL','bKStf','6WY1zk5','KqA6o','rYLxo','50oq7R3','Gi0jc','5Nk2p1P','v15Ie','gYg8RaO',
            #Ecuador
            'wbxM3BP','rkjlGvw','NGVdLzp','rnrlo','V7XPZ','4VpTMA9','sTUpBCS','Znune','2VuKN','hz9UY',
            #Germany
            'zrPRVuN','Ajf9g3D','lceHsT6','BAVZRRs','iHnbahA','XInPbu9','FDZ3lTW','NzeAAhx','HSqPaLA','1MZ2Z7l',
            #South Africa
            'Afjkw', 'xTUTYoE','m1Wg9Vt','E4ipiqV','08qa6','hlQI0','B36odyf','UTGGlNO','Aeh8MFi','cou0N',
            #Netherlands
            'Yuoe5UZ','Tn0f83g','WixNSG5','atkml','hftKF','9MPH3','jwiI9mp','t90HHoe','FdMalAQ','hRfH0kg',
            #Panama
            'vkIXqNa','30JQ5c4','6YA6xWo','XwSOJLw','SFBUW08','PvDi8','eok2Exi','RRPevbh','c2GTjSv','OshTmjE',
            #Argentina
            'Y6qMZTX','zkahmbF','PcO48fK','p8iAlVe','a1C8u','vmIi3','xsBKJJ5','SJ8Lo','cdd4N62','L1OxjHi',
            #Albania
            '7yDE0', 'bicsU', 'ZqCU9wv', '9A5rQXR', 'pUNN7', 'DR2an7h','V3dW4Dn', 'ePVGARS','TzZocYt','2NambtX',
            #Austria
            'DNF4IWn', 'mMxSAT6', 'MAw6Njp', 'CNVj2p5', 'A16u9pS', '1PxpZ','yxczE', 'EUxZpTv','joBhaAt','fgb1NW8',
            #Cameroon
            'lFScS0P', '0Ws36', 'UORFe6y', 'uYJlU0O', 'C0w9u4L', 'iG7yAv4','lGpGyyG', 'rYiW8yn','6efRLjr','na3XsN8',
            #Chile
            'eEmoAu9', '5lJ5myr', 'ZN6D4mV', 'G0jK5iu', 'J8a4kXv', 'Mg9Ma2y','wx4VIGt', '7xpzcop','5cIntm3','E4ALLZr',
            #Congo
            'jpGd7TF', '6NhDNUB', 'oEawfHo', 'NOM2vxP', '3MJxF5e', 'azlj2ma','fqqdVPR', '0XQ3MY9','10HQj2K','ElmA3W6',
            #Costa Rica
            'D64oGRw', 'EySdGX7', 'qI6qGOJ', 'EqRsr', '41Y0z7c', '0ox974Z','RteihhY', 'ARvLUw2','f8NBx6l','GJH0o',
            #Croatia
            'ldOMnZh', 'mBpgfPo', 'QqntK', 'sOnV0Xy', 'TWYTQ', 'Tx26sF8','KJmr4', 'Ylvxt','xmcvF','hHjflRD',
            #Nigeria
            '12vTZKZ', 'IvUl80S', 'NjCzt', 'kAZVc', 'Mz450Xr', 'efonoBH','ZfZTs4W', 'bQvrNv3','JyA7N','inHFemy',
            #Portugal
            'H7nvQFo', 'fYYulaI', 'kYSWkmD', 'HyJkbMv', '2xXjL', 'QhjBRh9','kS02H9B', '82yaC9e','p3lodH8','qSBRKl7',
            #Mongolia
            'njigrJo', 'xenUl', 'SZeOu', '62fI8Nf', '1YhE0Wq', '6yplsCo','HLvYPjk', 'MmIT0wE','a8qifsj','F7sDyzs',
            #Kazakhstan
            'wk9XARA', 'lACN4if', 'mETwduJ', 'Og6eUW0', 'IaV1fW9', 'ATb31Dj', 'aHh5Kfk', 'h7bTe', 'RrNX7', 'm9QsAQh',
            #Libya
            'XSC7wVS', 'HmY2zGp', 'xCbFj', 'ChMmPpY', 'o6BjCHi', '3dlQB8u', 'tobnp', 'cI91tHA', 'VxL4sdt', 'cK47uRK',
            #Madagascar
            'n8iaOQw', 'IkCxSBX', '5zJ0T', 'N0LPc', 'wnboVGI', 'IjWju', 'xnWps3C', '7Z0YR7P', 'v62sS', 'SsYSw4z',
            #Jamaica
            '6U5SPx6', 'ZvxtR1v', 'wlPK7', 'onTB5vb', 'oezeYg4', '9cY508M', 'ub59FKa', 'e8FmRwp', 'Li5GVjt', 'ijzsRU1',
            #Indonesia
            'YiQ9u', 'FklkvSE', 'A8EuJbV', 'ULGpV', 'GLS5sCL', 'HjrlFFC', 'aIrDEXH', 'SXp7RmY', 'GjC1EIC', '5jCE0',
            #Cambodia
            'WUsYuoc', 'YsR0y', '7X6DZ', 'RagHleg', 'mNsZZfh', 'KYr3QgU', 'Q5I1F', 'm2TvHjv', 'HGnho', 'RdQVSBB',
            #Guatemala
            'xYGcnRd', '20pjOv5', 'oIEczVw', 'AavPJ', 'YPwRZWv', 'hwdyO', 'GFaDI', 'ItHLUi5', '0IMK9k1', 'z9YPKuz',
            #Greece
            'cndVGQd', 'kEPmfaM', 'UtQgTH7', 'rA59e49', 'ViFMnGO', 'lYUup', 'X6E86', 'tfs1Gp9', 'IgIHvF5', 'i0mKUbv',
            #Canda
            #'', '', '', '', '', '','', '','','',
            #Canda
            #'', '', '', '', '', '','', '','','',
            #Canda
            #'', '', '', '', '', '','', '','','',
            #Canda
            #'', '', '', '', '', '','', '','','',
            #Canda
            #'', '', '', '', '', '','', '','','',
            ]

pictureId = ['DVNWyG8','7vvnSCM','YGAxqzh']

def getGalleryInfo(pictureId):
    #https://code.i-harness.com/fr/q/d59bc4
    header= {"Content-Type": "text", "Authorization": "Client-ID " + imgur_client}
    r = requests.get("https://api.imgur.com/3/gallery/"+pictureId, headers=header)
    j = json.loads(r.text)
    return j['data']

def getImgInfo(pictureId):
    #https://code.i-harness.com/fr/q/d59bc4
    header= {"Content-Type": "text", "Authorization": "Client-ID " + imgur_client}
    r = requests.get("https://api.imgur.com/3/image/"+pictureId, headers=header)
    j = json.loads(r.text)
    return j['data']

def getImgComments(gallery_id):
    #https://code.i-harness.com/fr/q/d59bc4
    header= {"Content-Type": "text", "Authorization": "Client-ID " + imgur_client}
    #r = requests.get("https://api.imgur.com/3/image/"+gallery_id+"/comments/", headers=header)
    r = requests.get("https://api.imgur.com/3/gallery/"+gallery_id+"/comments/", headers=header)
    j = json.loads(r.text)

    return j['data']
    #comments = [x['comment'] for x in j['data']]
    #return comments

def Gallery_Description_DataFrame(galeries):
    df = pd.DataFrame(columns=['gallery_id','link','description','views','downs','ups','favorite_count','section','tags','title','topic','images_count','images'])
    for gallery_id in galeries:
        Data = getGalleryInfo(gallery_id)
        print(gallery_id)
        df.loc[df.shape[0]] = [Data['id'],Data['link'],Data['description'],Data['views'],Data['downs'],Data['ups'],Data['favorite_count'],Data['section'],
                              [tag['description'] for tag in Data['tags'] if tag['description']!=''],Data['title'],Data['topic'],1,[]]
        if 'images' in Data.keys():
           df.at[df.shape[0]-1, 'images'] = [image['id'] for image in Data['images']]
           df.at[df.shape[0]-1, 'images_count'] = Data['images_count'] 
           df.at[df.shape[0]-1, 'link'] = [image['link'] for image in Data['images']][0]
    df.to_csv('Galeries.csv', sep=',',encoding='utf-8')
    return df

def Image_Description_DataFrame(dfG):
    df = pd.DataFrame(columns=['gallery_id','image_id','link','description','views','section','tags','title','google_Vision','Microsoft_CaptionBot', 'JasonBrownlee','comments'])
    i = 0
    for image_ids in dfG.images.tolist():
        for image_id in image_ids:
            Data = getImgInfo(image_id)
            df.loc[df.shape[0]] = [dfG.at[i,'gallery_id'],Data['id'],Data['link'],Data['description'],Data['views'],Data['section'],[tag['description'] for tag in Data['tags'] if tag['description']!=''],Data['title'],'','','',getImgComments(dfG.at[i,'gallery_id'])]
        i+=1
    return df

def get_Replies(children):
    replies = []
    if len(children) > 0:
        for child in children:
            replies.extend(x['comment'] for x in child['children'])
            replies.extend(get_Replies(child['children']))
    #replies.append(comment['comment'])
    return replies

def Comments_Replies_DataFrame(gallery_id):
    Data = getImgComments(gallery_id) 
    df = pd.DataFrame(columns=['image_id','gallery_id','comment_id','comment','downs', 'ups','replies'])

    i = 0
    for comment in Data:
        replies = get_Replies(comment['children'])
        print (replies)
        df.loc[df.shape[0]] = [gallery_id,gallery_id,comment['id'],comment['comment'],comment['downs'],comment['ups'],replies]
        i+=1 
    #df = pd.DataFrame({'gallery_id':'','image_id':'','title': ImgInf['title'],'google_Vision':'','Microsoft_CaptionBot' : '', 'JasonBrownlee': '', 'comments': ImgComnts, 'replies': ''})
    return df

def Download_image(image_url, download_path):
    'https://jloh.co/posts/downloading-imgur-albums-with-python/'
    try:
        #download = urllib.request.URLopener()
        #download.retrieve(image_url, download_path)

        with open(download_path, 'wb') as f:
            f.write(requests.get(image_url).content)

        print("File {} downloaded to {}".format(image_url, download_path))

    except urllib.error.URLError as e:
        print("Error downloading image '{}': {}".format(image_url, e))
    except urllib.error.HTTPError as e:
        print("HTTP Error download image '{}': {!s}".format(image_url, e))
    
def Download_DataSet():
    onlyImage = []
    DataSet = '/home/polo/.config/spyder-py3/PhD/Tourism50'
    try:  
        os.mkdir(DataSet)
    except OSError:  
                print ("Creation of the directory %s failed" % DataSet)
   
    for Country in Countries:
        try:  
            os.mkdir(DataSet+'/'+Country)
        except OSError:  
                print ("Creation of the directory %s failed" % Country)
        
    for gallery_id in galeries:
        Data = getGalleryInfo(gallery_id)

        if 'images' in Data.keys():
            try:
                os.rmdir(DataSet+'/'+gallery_id)
            except OSError:  
                    print ("Deletion of the directory %s failed" % gallery_id)
            #images = Data['images']
            #for image in images:
            #    Download_image(image['link'], DataSet+'/'+gallery_id+'/'+image['id'])#+'.jpg')
        else:
            print('yazid id '+gallery_id)
            try:
                os.mkdir(DataSet+'/'+gallery_id)
            except OSError:  
                print ("Creation of the directory %s failed" % gallery_id)
                
            onlyImage.append(gallery_id)
            #print(Data['link'])
            #return Data

            Download_image(Data['link'], DataSet+'/'+Country+'/'+gallery_id+'/'+Data['id']+'.jpg')
        break
    print (onlyImage)

def AllThing(gallery_id):
    Gkeys = ['id','link','description','datetime','downs','ups','favorite_count','section','tags','title','topic','images']
    Data = getGalleryInfo(gallery_id)
    Ginf = {x: Data[x] for x in Gkeys if x in Data}
    
    if 'images' not in Ginf.keys():
        Ginf['Comments'] = getImgComments(gallery_id)
    else:
        Ginf['Comments'] = getImgComments(Ginf['images'][0]['id'])

    return Ginf
    
def Download_DataSetFromCSV():
    #dfG = pd.read_csv('Galeries.csv', sep=',',encoding='utf-8')
    DataSet = '/home/polo/PhD NLP/PhD_Dataset'
    DataSet = '/home/polo/.config/spyder-py3/PhD/Tourism48'

    try:  
        os.mkdir(DataSet)
    except OSError:  
                print ("Creation of the directory %s failed" % DataSet)
   
    #links_Matrix = np.array(dfG.link.tolist()).reshape(len(Countries),10)
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)
    i = 0
    for Country in Countries:
        try:  
            os.mkdir(DataSet+'/'+Country)
            for j in range (10):
                os.mkdir(DataSet+'/'+Country+'/'+Galeries_Matrix[i,j])
                Data = AllThing(Galeries_Matrix[i,j])
                with open(DataSet+'/'+Country+'/'+Galeries_Matrix[i,j]+'/'+Galeries_Matrix[i,j]+'.json', 'w') as outfile:
                     json.dump(Data, outfile)
                Download_image(Data['link'], DataSet+'/'+Country+'/'+Galeries_Matrix[i,j]+'/'+Data['id']+'.jpg')     
                #Download_image(links_Matrix[i,j], DataSet+'/'+Country+'/'+Galeries_Matrix[i,j]+'/'+links_Matrix[i,j][links_Matrix[i,j].rindex('/')+1:])
                with open(DataSet+'/'+Country+'/'+Galeries_Matrix[i,j]+'/'+'google.json', 'w') as f:
                     f.write('google')
        except OSError as e:  
                print ("Creation of the directory %s failed" % Country)
        i += 1
        

#______________________________________________________________________________

#Data = Download_DataSet()

#Download_DataSetFromCSV()

Data = AllThing('EUxZpTv')