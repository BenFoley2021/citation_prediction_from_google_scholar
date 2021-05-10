# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:53:53 2021

@author: Ben
"""
import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from bs4 import BeautifulSoup

import pickle


login_info = pickle.load(open('login_info.pickle', 'wb'))


driver = webdriver.Chrome('./chromedriver')

driver.get("https://accounts.google.com/signin")

email = driver.find_element_by_name("identifier")

email.send_keys(login_info[0])
