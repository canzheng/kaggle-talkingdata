#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 18:46:49
# @Author  : Can Zheng (can.zheng@gmail.com)


import sms
import sys
tpl_value = {'#name#': sys.argv[1]}
data = sms.tpl_send_sms(tpl_value=tpl_value)
print(data.decode("utf-8", "ignore"))
