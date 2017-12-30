import nltk

# 本文，大意是歐巴馬卸任
news_content='''At noon on Friday, 55-year old Barack Obama became a federal retiree.
His pension payment will be $207,800 for the upcoming year, about half of his presidential salary.
Obama and every other former president also get seven months of "transition" services to help adjust to post-presidential life. The ex-Commander in Chief also gets lifetime Secret Service protection as well as allowances for things such as travel, office expenses, communications and health care coverage.
All those extra expenses can really add up. In 2015 they ranged from a bit over $200,000 for Jimmy Carter to $800,000 for George W. Bush, according to a government report. Carter doesn't get health insurance because you have to work for the federal government for five years to qualify.
'''

# 分詞、標註、NER、打分數，依分數高低排列句子
results=[]
for sent_no,sentence in enumerate(nltk.sent_tokenize(news_content)):
    no_of_tokens=len(nltk.word_tokenize(sentence))
    # Let's do POS tagging
    tagged=nltk.pos_tag(nltk.word_tokenize(sentence))
    # Count the no of Nouns in the sentence
    no_of_nouns=len([word for word,pos in tagged if pos in ["NN","NNP"] ])
    #Use NER to tag the named entities.
    ners=nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)), binary=False)
    no_of_ners= len([chunk for chunk in ners if hasattr(chunk, 'label')])
    score=(no_of_ners+no_of_nouns)/float(no_of_tokens)
    results.append((sent_no,no_of_tokens,no_of_ners, no_of_nouns,score,sentence))

# 依重要性順序列出句子
for sent in sorted(results,key=lambda x: x[4],reverse=True):
    print(sent[5])
