import requests
from requests.exceptions import SSLError
from bs4 import BeautifulSoup

txt_path = 'RawText__xinhua.txt'

def Paqu(url):
    # 通过访问新华网主页导航爬取各个分类新闻index
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有的<a>标签，并获取其href属性
    links = []
    for a_tag in soup.find_all('a', href=True):
        link = a_tag['href']
        text = a_tag.get_text(strip=True)  # 获取链接文本
        if link.startswith("http") and any(char.isdigit() for char in link) and text:  # 筛选特定链接
            links.append(link)

    with open(txt_path, 'a', encoding='utf-8') as txt_file:
        for url_news in links:
            try:
                res = requests.get(url_news, timeout=10)
                res.encoding = res.apparent_encoding
            except SSLError as e:
                print(f"SSL error while fetching {url_news}: {e}")
            except requests.RequestException as e:
                print(f"Request failed for {url_news}: {e}")

            # 根据网站的编码方式修改解码方式，因为网页的文字编码方式多种多样有UTF-8 GBK这些解码方式如果不一致容易发生乱码，所以解码方式最好不要统一，而是交给网页自己来决定
            soup_new= BeautifulSoup(res.text, 'html.parser')  # 使用html5lib样式来解析网页
            # 查找所有的<p>标签，并获取其href属性
            data = soup_new.select('p')  # 元素选择器
            news_text = ''
            paragraphs = [p.text.strip() for p in data]
            news_text = '\n'.join(paragraphs) + '\n'
            # 目的是全部新闻都写到文件中
            txt_file.write(news_text)
    print(url + '已经被爬取')

if __name__ == '__main__':
    # 通过访问新华网主页导航爬取各个分类新闻连接
    url_Xinhua = 'http://www.news.cn'
    response = requests.get(url_Xinhua)
    soup = BeautifulSoup(response.text, 'html.parser')

    Links = {}
    for el in soup.find(class_="nav-item nav-top"):
        Links[el.text] = el['href']

    for el in soup.find(class_="nav-item nav-bottom"):
        Links[el.text] = el['href']


    # 调用paqu函数爬取每个分类的各条新闻段落内容
    for url in Links.values():
        Paqu(url)

