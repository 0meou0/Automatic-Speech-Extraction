# -*- coding: utf-8 -*-
# project1模块

from app import BaseHandler


class Project1(BaseHandler):


    def find_tile(self, text):
        return text[:3]


    def get(self, *args, **kwargs):
        # 发送数据到前端
        data = [1,2,3,4,5]
        self.render('project1.html', data=data, news=None)


    def post(self, *args, **kwargs):
        say_word = ['指出', '说', '表示', '声称', '说道', '宣称', '告诉', '提到', '认为', '写道', '相信', '称', '说明', '否认', '透露', '强调', '指称',
                '表明', '提及', '提出', '问', '觉得', '回答', '说出', '暗示', '辩称', '坚称', '谈到', '断言', '解释']
        # 接收前端输入self.new存储前端name为news的内容
        news = self.get_argument('news')
        # print(type(news))
        cleaned_string = self.token(news)
        # result = self.cut_sentence(cleaned_string)
        result = self.extract_comment(cleaned_string, say_word)
        # result = Project1.find_tile(self, news)
        # data:<list>
        self.render('project1.html', data=result, news=news)
