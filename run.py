# -*- coding: utf-8 -*-
from tornado import web, ioloop, httpserver, options
import tornado.autoreload
from app import project1,project2,project3


# 功能模块
class MainPageHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        # self.write('helloworld')
        self.render('index.html')

    def post(self, *args, **kwargs):
        news = self.get_argument('news')
        print('news:{}'.format(news))




class Project2(web.RequestHandler):
    def get(self, *args, **kwargs):
        self.render('project2.html')


#
settings = {
    # 设置模板路径
    'template_path': 'app/templates',
    # 设置静态文件路径
    'static_path': 'app/static'
}

# 路由
application = web.Application([
    (r"/index", MainPageHandler),
    (r"/project1", project1.Project1),
    (r"/project2", project2.Project2),
], **settings)

if __name__ == '__main__':
    # socket
    http_server = httpserver.HTTPServer(application)
    http_server.listen(9999)
    print('http//:127.0.0.1:9999')

    ioloop.IOLoop.current().start()
