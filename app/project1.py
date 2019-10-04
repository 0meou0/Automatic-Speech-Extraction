# -*- coding: utf-8 -*-
# project1模块

from app import BaseHandler


class Project1(BaseHandler):

    def get(self, *args, **kwargs):
        # 发送数据到前端
        self.render('project1.html', data=None, news=None)

    def post(self, *args, **kwargs):
        say_word = ['指出', '说', '表示', '声称', '说道', '宣称', '告诉', '提到',
                    '认为', '写道', '相信', '称', '说明', '否认', '透露', '强调', '指称',
                    '表明', '提及', '提出', '问', '觉得', '回答', '说出', '暗示', '辩称',
                    '坚称', '谈到', '断言', '解释', '谈到', '谈话', '讲到', '宣告', '宣布',
                    '眼中', '指出', '坦言', '明说', '报道', '通知',
                    '看来', '所说', '透露', '眼里', '直言',
                    '反问', '咨询', '发言', '反映', '谈论', '谴责',
                    '批评', '抗议', '反对', '申诉', '狡辩', '重申',
                    '通报', '通报', '询问', '正说', '介绍'
                    ]
        # 接收前端输入self.new存储前端name为news的内容
        news = self.get_argument('news')
        # 去除无意义字符
        cleaned_string = self.token(news)
        # result = self.split_sentence(cleaned_string, say_word, ratio=0.05)
        result = self.extract_comment(cleaned_string, say_word)
        # data:<list>
        self.render('project1.html', data=result, news=news)
