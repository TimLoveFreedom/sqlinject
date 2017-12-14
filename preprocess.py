__author__ = 'jellyzhang'
import pylibinjection

#negative
with open('raw_negative.txt','r',encoding='utf-8',errors='ignore')as fread,open('negative.txt','w',encoding='utf-8',errors='ignore')as fwrite:
    for line in fread:
        sqli_result=pylibinjection.detect_sqli(bytes(line.rstrip(), encoding='utf-8'))
        fingerprint=str(sqli_result['fingerprint'],encoding='utf-8')
        fwrite.write('{}\n'.format(fingerprint))




#positive
with open('raw_positive.txt','r',encoding='utf-8',errors='ignore')as fread,open('positive.txt','w',encoding='utf-8',errors='ignore')as fwrite:
    for line in fread:
        sqli_result = pylibinjection.detect_sqli(bytes(line.rstrip(), encoding='utf-8'))
        fwrite.write('{}\n'.format(str(sqli_result['fingerprint'],encoding='utf-8')))
