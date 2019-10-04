# -*- coding: utf-8 -*-
from tornado.web import RequestHandler, Finish
import re
import jieba
import os
from pyltp import Segmentor, SentenceSplitter, NamedEntityRecognizer, Parser, Postagger

from typing import List, Dict
import difflib

import math
from itertools import product, count
from string import punctuation
from heapq import nlargest


# # ltp路径
# LTP_DATA_PATH = 'D:\pyltp-master\ltp_data_v3.4.0'
#
# cws_model_path = os.path.join(LTP_DATA_PATH, 'cws.model')
# pos_model_path = os.path.join(LTP_DATA_PATH, 'pos.model')
# ner_model_path = os.path.join(LTP_DATA_PATH, 'ner.model')
# par_model_path = os.path.join(LTP_DATA_PATH, 'parser.model')
#

class BaseHandler(RequestHandler):
    """
    基类
    """

    def jieba_cut(self, string):
        return " ".join(jieba.cut(string))

    def token(self, string):
        pat = re.compile('\\\\n|\\u3000|;|\\n|\s+')
        string = re.sub(pat, '', string)
        return ''.join(string)

    def cut_sentence(self, string):
        """
        分句
        :param string: atricles
        :return: list[sentence1,sentence2...]
        """
        sentence_cut = SentenceSplitter.split(string)
        sentences = [s for s in sentence_cut if len(s) != 0]
        return sentences

    def cut_word(self, sentence):
        """
        分词（jieba）
        :param sentence: list[sentence1,sentence2...]
        :return: list[word1,word2...] words:[word1,word2...]
        """
        words = []
        words += self.jieba_cut(sentence).split(' ')
        return words

    def word_pos(self, sentence, postagger):
        """
        词性标注
        :param sentence:list[sentence1,sentence2...]
        :return: list[postag1,postag2...]
        """
        words = self.cut_word(sentence)
        postag = postagger.postag(words)
        return list(postag)

    def ner(self, words, pos, recognizer):
        """
        命名实体识别
        :param words:cut_word_list
        :param pos: postag_list
        :return: ner_list
        """
        ners = recognizer.recognize(words, pos)
        return list(ners)

    def dependency_parse(self, words, pos, parser):
        """
        依存句法分析
        :param words:cut_word_list
        :param pos: pos_list
        :return: arc.head:依存关系头索引,arc.relation:依存关系
        """
        arcs = parser.parse(words, pos)
        return [(arc.head, arc.relation) for arc in arcs]

    def find_str_index(self, source_list, begin_index, target_list):
        """
        在分好词的列表中查找指定字符之一
        :param source_list: 要查找的列表
        :param begin_index: 开始位置索引
        :param target_list: 目标字符列表，可按重要程度排序
        :return: 位置 失败返回-1
        """
        for item in target_list:
            res = [i for i in range(len(source_list)) if source_list[i] == item and i >= begin_index]
            if len(res) != 0:
                return res[0]
            else:
                continue
        else:
            print('没找到{}'.format(target_list))
            return -1

    def extract_comment(self, article, say_words):
        """
        抽取言论
        :param news_path: 新闻路径
        :param say_words: similar to "say"
        :return:result:list[[person, say, comment],...]
        """
        # ltp路径
        LTP_DATA_PATH = 'D:\pyltp-master\ltp_data_v3.4.0'

        cws_model_path = os.path.join(LTP_DATA_PATH, 'cws.model')
        pos_model_path = os.path.join(LTP_DATA_PATH, 'pos.model')
        ner_model_path = os.path.join(LTP_DATA_PATH, 'ner.model')
        par_model_path = os.path.join(LTP_DATA_PATH, 'parser.model')

        postagger = Postagger()
        postagger.load(pos_model_path)
        print('Postagger loaded!')
        recognizer = NamedEntityRecognizer()
        recognizer.load(ner_model_path)
        print('NER loaded!')
        parser = Parser()
        parser.load(par_model_path)
        print('Parser loaded!')

        result = []
        sentences = self.cut_sentence(self.token(article))
        for s_index, sentence in enumerate(sentences):
            words = self.cut_word(sentence)
            pos = self.word_pos(sentence, postagger)
            ner_list = self.ner(words, pos, recognizer)
            parse_list = self.dependency_parse(words, pos, parser)
            if 'S-Nh' or 'S-Ni' or 'S-Ns' in ner_list:
                comment = ''
                for p_index, p in enumerate(parse_list):
                    # p[0]-1:说的索引（words，parse_list中都是）
                    # p_index:主语位置

                    if (p[1] == 'SBV') and words[p[0] - 1] in say_words:
                        say = words[p[0] - 1]
                        person = words[p_index]
                        p_i = 1
                        while p_i <= p_index and parse_list[p_index - p_i][1] == 'ATT':
                            person = words[p_index - p_i] + person
                            p_i = p_i + 1
                        # 说后是。找前一句话的“”
                        if words[p[0]] == '。':
                            # print('说。')
                            i = 1
                            last_sentence = sentences[s_index - i]
                            last_words = self.cut_word(last_sentence)
                            begin = self.find_str_index(last_words, 0, ['“'])
                            end = self.find_str_index(last_words, 0, ['”'])
                            if begin != -1 and end != -1 and begin < end:
                                comment = ''.join(last_words[begin + 1:end])
                            else:
                                while begin == -1 and end != -1:
                                    i = i + 1
                                    last_sentence = sentences[s_index - i]
                                    last_words = self.cut_word(last_sentence)
                                    begin = self.find_str_index(last_words, 0, ['“'])
                                while i > 0:
                                    comment = comment + sentences[s_index - i]
                                    i = i - 1
                        else:
                            begin = self.find_str_index(words, p[0], ['“'])
                            end = self.find_str_index(words, p[0], ['”'])
                            if begin != -1 and end != -1 and parse_list[end - 1][0] == 'WP':
                                comment = ''.join(words[begin:end])
                            elif begin != -1 and end == -1:
                                comment = ''.join(words[begin:])
                                i = 1
                                next_sentence = sentences[s_index + i]
                                while end == -1:
                                    end = self.find_str_index(self.cut_word(next_sentence), 0, ['”'])
                                    i = i + 1
                                    if len(sentences) > s_index + i:
                                        next_sentence = sentences[s_index + i]
                                    else:
                                        break
                                comments = ''
                                while i > 1 and len(sentences) > s_index + i:
                                    comments = sentences[s_index + i] + comments
                                    i = i - 1
                                comment = comment + comments

                            else:
                                # 说后面跟，或：
                                if words[p[0]] == ',' or words[p[0]] == '，' or words[p[0]] == ':':
                                    # print('说，')
                                    end = self.find_str_index(words, p[0] + 1, ['。', '！'])
                                    if end != -1:
                                        comment = ''.join(words[p[0] + 1:end])
                                        # 说后跟宾语
                                elif parse_list[p[0]][1] == 'VOB' or parse_list[p[0]][1] == 'IOB':
                                    print('告诉谁')
                                    i = 0
                                    while len(comment) == 0:
                                        end = self.find_str_index(words, p[0] + i, ['，', '。', '！'])
                                        if end != -1:
                                            comment = ''.join(words[p[0] + i:end])
                                        i = i + 1
                                        # 说后面直接跟内容
                                else:
                                    # print('说内容')
                                    end = self.find_str_index(words, p_index, ['，', '。', '！'])
                                    if end != -1:
                                        comment = ''.join(words[p[0]:end])
                        print(parse_list)
                        # print(words[p[0]])
                        print(sentence)
                        print('[{}] [{}] [{}]'.format(person, say, comment))
                        print('-' * 50)
                        item = []
                        # item.append(person)
                        # item.append(say)
                        # item.append(comment)
                        result.append([person, say, comment])
                        # result.append(item)

        postagger.release()
        recognizer.release()
        parser.release()

        return result


    def split_sentence(self, sentence=None, say_word_list: List[str] = None,
                       cycle: bool = True, ratio: float = None) -> None:
        """
        分词
        :type say_word_list:
        :param sentence:
        :return:
        """
        LTP_DATA_PATH = 'D:\pyltp-master\ltp_data_v3.4.0'

        cws_model_path = os.path.join(LTP_DATA_PATH, 'cws.model')
        pos_model_path = os.path.join(LTP_DATA_PATH, 'pos.model')
        ner_model_path = os.path.join(LTP_DATA_PATH, 'ner.model')
        par_model_path = os.path.join(LTP_DATA_PATH, 'parser.model')

        postagger = Postagger()
        postagger.load(pos_model_path)
        print('Postagger loaded!')
        parser = Parser()
        parser.load(par_model_path)
        print('Parser loaded!')
        segment = Segmentor()
        segment.load(cws_model_path)
        print('CWS loaded!')
        if cycle == True:

            try:
                lines = sentence
                sentence = list(segment.segment(lines))
                # print('sen ok')
                # 找出相似
                find_say_word = [word for word in sentence if word in say_word_list]
                if len(find_say_word) == 0:
                    print('没有发现类似“说”的单词!')
                else:
                    post_word = postagger.postag(sentence)
                    post_word = list(post_word)
                    # print('post ok')
                    parse_word = parser.parse(sentence, post_word)
                    parse_word = [(arc.head, arc.relation)
                                  for arc in parse_word]

                    # print('parse ok')
                    counter_index = 0
                    for index, word in enumerate(parse_word):
                        location_part1 = ''
                        location_part2 = ''
                        location_part3 = ''
                        # 找出第一个SBV下的"真新闻"
                        if word[-1] == 'SBV':
                            counter_index = word[0]
                            location_part1 += sentence[index]
                            location_part1 += sentence[word[0] - 1]
                            break
                    # 先将整个SBV后面碰到是双引号或者没有双引号的句子,用于后面文本向量的模型计算
                    # 暂时只提取双引号内容和两个句号结束的句子为数据
                    if sentence[counter_index] == '"':
                        for index_2, word_2 in enumerate(sentence[counter_index + 1:]):
                            if word_2 == '"':
                                break
                            location_part2 += word_2
                    else:
                        for index_2, word_2 in enumerate(sentence[counter_index:]):
                            if word_2 == '。':
                                for word_4 in sentence[index_2 + 1:]:
                                    if word_4 == '。':
                                        break
                                    location_part3 += word_4
                                break
                            location_part2 += word_2
                    # 判别说前后两个句号句子的相似度
                    cal_ratio = difflib.SequenceMatcher(None, location_part2, location_part3).ratio()
                    if cal_ratio > ratio:
                        result = location_part1 + location_part2 + location_part3
                    else:
                        result = location_part1 + location_part2
                segment.release()
                postagger.release()
                parser.release()
                return result.strip('\n')
            except Exception as e:
                print(e)

        elif cycle == False:
            print('不处理')
        else:
            raise TypeError('错误的输入类型')
        print('词标注和上下文定义结束')
        print('-' * 20, '华丽的分割线', '-' * 20)


# project2
class BaseHandler2(RequestHandler):
    # TEXTPAGE
    # def __init__(self, sentence):
    #         self.__sentence = sentence

    def token(self, string):
        pat = re.compile('\\\\n|\\u3000|;|\\n|\s+')
        string = re.sub(pat, '', string)
        return ''.join(string)

    def _calculate_similarity(self, sen1, sen2):
        counter = 0
        for word in sen1:
            if word in sen2:
                counter += 1
        return counter / (math.log(len(sen1)) + math.log(len(sen2)))

    # 构造有向图
    def _create_graph(self, word_sent):
        num = len(word_sent)

        board = [[0.0 for _ in range(num)] for _ in range(num)]

        for i, j in product(range(num), repeat=2):
            if i != j:
                board[i][j] = self._calculate_similarity(word_sent[i], word_sent[j])
        return board

    def _weighted_pagerank(self, weight_graph):
        """
            输入相似度邻接矩阵
            返回各个句子的分数
            """
        # 把初始的分数值设置为0.5
        scores = [0.5 for _ in range(len(weight_graph))]
        old_scores = [0.0 for _ in range(len(weight_graph))]

        # 开始迭代
        while self._different(scores, old_scores):
            for i in range(len(weight_graph)):
                old_scores[i] = scores[i]

            for i in range(len(weight_graph)):
                scores[i] = self._calculate_score(weight_graph, scores, i)
        return scores

    def _different(self, scores, old_scores):
        """
            判断前后分数有没有变化
            这里认为前后差距小于0.0001
            分数就趋于稳定
        """
        flag = False
        for i in range(len(scores)):
            if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
                flag = True
                break
        return flag

    def _calculate_score(self, weight_graph, scores, i):
        """
            根据公式求出指定句子的分数
        """
        length = len(weight_graph)
        d = 0.85
        added_score = 0.0

        for j in range(length):
            fraction = 0.0
            denominator = 0.0
            # 先计算分子
            fraction = weight_graph[j][i] * scores[j]
            # 计算分母
            for k in range(length):
                denominator += weight_graph[j][k]
            added_score += fraction / denominator
        # 算出最终的分数
        weighted_score = (1 - d) + d * added_score

        return weighted_score

    def Summarize(self, n, sentence):
        # 首先分出句子
        # sents = sent_tokenize(text)
        sentence = sentence.split('\n')
        sents = []
        print('sentence:', sentence, type(sentence))
        for line in sentence:
            sents = re.split('[。？！]', line)[:-1]

        # 然后分出单词
        # word_sent是一个二维的列表
        # word_sent[i]代表的是第i句
        # word_sent[i][j]代表的是
        # 第i句中的第j个单词
        word_sent = [list(jieba.cut(s)) for s in sents]
        print(word_sent)

        # 把停用词去除
        # for i in range(len(word_sent)):
        # for word in word_sent[i]:
        # if word in stopwords:
        # word_sent[i].remove(word)
        similarity_graph = self._create_graph(word_sent)
        scores = self._weighted_pagerank(similarity_graph)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(n):
            sent_index.append(sent_selected[i][1])
        return [sents[i] for i in sent_index]
