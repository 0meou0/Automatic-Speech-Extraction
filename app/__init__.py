# -*- coding: utf-8 -*-
from tornado.web import RequestHandler, Finish
import re
import jieba
import os
from pyltp import Segmentor, SentenceSplitter, NamedEntityRecognizer, Parser, Postagger

# ltp路径
LTP_DATA_PATH = 'D:\pyltp-master\ltp_data_v3.4.0'

cws_model_path = os.path.join(LTP_DATA_PATH, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_PATH, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_PATH, 'ner.model')
par_model_path = os.path.join(LTP_DATA_PATH, 'parser.model')


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

    def word_pos(self, sentence):
        """
        词性标注
        :param sentence:list[sentence1,sentence2...]
        :return: list[postag1,postag2...]
        """
        postagger = Postagger()
        postagger.load(pos_model_path)
        print('Postagger loaded!')

        words = self.cut_word(sentence)
        postag = postagger.postag(words)

        postagger.release()
        return list(postag)

    def ner(self, words, pos):
        """
        命名实体识别
        :param words:cut_word_list
        :param pos: postag_list
        :return: ner_list
        """
        recognizer = NamedEntityRecognizer()
        recognizer.load(ner_model_path)
        print('NER loaded!')

        ners = recognizer.recognize(words, pos)
        recognizer.release()
        return list(ners)

    def dependency_parse(self, words, pos):
        """
        依存句法分析
        :param words:cut_word_list
        :param pos: pos_list
        :return: arc.head:依存关系头索引,arc.relation:依存关系
        """
        parser = Parser()
        parser.load(par_model_path)
        print('Parser loaded!')

        arcs = parser.parse(words, pos)
        parser.release()
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
        result = []
        sentences = self.cut_sentence(self.token(article))
        for s_index, sentence in enumerate(sentences):
            words = self.cut_word(sentence)
            postagger = self.word_pos(sentence)
            ner_list = self.ner(words, postagger)
            parse_list = self.dependency_parse(words, postagger)
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

        return result
