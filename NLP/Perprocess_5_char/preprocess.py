test_file = 'RawText__xinhua.txt' #数据集
savepath='RawText__xinhua_Refine.txt'
with open(test_file,'r',encoding='utf-8')as f:
    words=f.read()
cutlist=['，','。','！','？','“','”','《','》','——','：','-','_',' ','、',' · ','\n','.',' ']
str=''
allwords=''
for word in words:
    if word in cutlist:
        pass
    elif len(str)==5:
        str+='\n'
        allwords+=str
        str=''
    else:
        str+=word
allwords+=str
print(allwords)
with open(savepath,'w',encoding='utf-8')as nf:
    nf.write(allwords)

