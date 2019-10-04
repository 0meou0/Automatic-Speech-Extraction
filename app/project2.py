# -*- coding: utf-8 -*-
from app import BaseHandler2


class Project2(BaseHandler2):
    def get(self, *args, **kwargs):
        # 发送数据到前端
        self.render('project2.html', data=None, news=None)

    def post(self, *args, **kwargs):
        # 接收前端输入self.new存储前端name为news的内容
        news = self.get_argument('news')
        cleaned_string = self.token(news)
        result = self.Summarize(3, cleaned_string)
        res = ''
        for sens in result:
            res += sens
            res += '。'
        # data:<str>
        self.render('project2.html', data=res, news=news)
