#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

from tornado.ioloop import IOLoop, PeriodicCallback
from tornado import gen
from tornado.websocket import websocket_connect


class ClientWebSocket(object):
    def __init__(self, url, camera, cars_detected):
        self.url = url
        # self.timeout = timeout
        self.camera = camera
        self.cars_detected = cars_detected

        self.ioloop = IOLoop.instance()
        self.ws = None
        self.connect()
        self.periodic = PeriodicCallback(self.keep_alive, 10000)  # 10 seconds
        self.periodic.start()
        # self.ioloop.start()

    @gen.coroutine
    def connect(self):
        try:
            self.ws = yield websocket_connect(self.url)
        except:
            print ("connection error")
        else:
            print ("connected")
            self.run()

    @gen.coroutine
    def run(self):
        while True:
            try:
                cars = []
                while not self.cars_detected.empty():
                    cars.append(self.cars_detected.get())
                if len(cars) > 0:
                    msg = json.dumps({'type': 'cars_detected', 'camera': self.camera, 'cars': cars})
                    self.ws.write_message(msg)
                yield gen.sleep(5)
            except:
                print ("connection closed")
                self.ws = None
                break

    def keep_alive(self):
        try:
            if self.ws is None:
                self.connect()
            else:
                self.ws.write_message(json.dumps({"type": "keep alive"}))
        except:
            print ("Error")
            self.ws = None
