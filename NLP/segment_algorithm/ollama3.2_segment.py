import ollama


# 流式输出
def api_generate(text: str):
    print(f'提问：{text}')

    stream = ollama.generate(
        stream=True,
        model='llama3.2',  # 修改大模型名称1
        prompt=text,
    )

    print('-----------------------------------------')
    for chunk in stream:
        if not chunk['done']:
            print(chunk['response'], end='', flush=True)
        else:
            print('\n')

def read_text(file_path):
    """
    读取文本文件，返回文本列表
    """
    texts = ''
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text = line.replace(" ", "")
            if text:
                texts += text
    return texts

if __name__ == '__main__':

    # textfile = '../Data/测试库10000字.txt'
    # text = read_text(textfile)
    # 流式输出
    raw_text = ''
    api_generate(text=f'请你帮我做一个分词的任务，用空格隔开每个词，只用给我回复结果,原文本：{raw_text}')

    # api_generate(text=text)

    # # 非流式输出
    # content = ollama.generate(model='llama3.2', prompt='天空为什么是蓝色的？')  # 修改大模型名称2
    # print(content)