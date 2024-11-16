---
title: Web Crawler for eCampus by Selenium
date: 2020-04-19 04:40:15
categories:
- Etc.
tags:
- Python
- Selenium
- B.S. Course Work
---
# 코로나 싫어서 대출하는 프로그램. .

<!-- More -->

~~~Python
from selenium import webdriver
import time


def print_of_lec():
    num = browser.find_elements_by_xpath('//*[@id="contentsIndex"]/div[2]/div[2]/ol/li/em')
    num = len(num)
    for i in range(num):
        lec_path = '//*[@id="contentsIndex"]/div[2]/div[2]/ol/li[' + str(i+2) + ']/em'
        lec_name = browser.find_elements_by_xpath(lec_path)
        for lec in lec_name:
            name = lec.text
        print(str(i+1) + '.\t' + name)


def start_lec(num, week): # num 번째 수업, week 주차
    lec_path = '//*[@id="contentsIndex"]/div[2]/div[2]/ol/li[' + str(num + 1) + ']/em'
    browser.find_element_by_xpath(lec_path).click()
    week_path = '//*[@id="week-' + str(week) + '"]'
    browser.find_element_by_xpath(week_path).click()
    return(browser.current_url)


def tim_to_sec(tim):
    tr = 0
    if(len(tim)==7):
        tr = int(tim[5] + tim[6]) + 60 * int(tim[2] + tim[3]) + 60 * 60 * int(tim[0])
    elif(len(tim)==5):
        tr = int(tim[3] + tim[4]) + 60 * int(tim[0] + tim[1])
    elif(len(tim)==4):
        tr = int(tim[2] + tim[3]) + 60 * int(tim[0])
    elif(len(tim)==2):
        tr = int(tim[0] + tim[1])
    elif(len(tim)==1):
        tr = int(tim[0])
    else:
        print('Error')
        print(tim)
    tr = tr + 100
    return(tr)


def play_lec(backward):
    time.sleep(1)
    NofLec = len(browser.find_elements_by_xpath('/html/body/div[3]/div[2]/div/div[2]/div[2]/div[3]/div'))
    for i in range(NofLec):
        browser.get(backward)
        seltd_lec = '/html/body/div[3]/div[2]/div/div[2]/div[2]/div[3]/div[' + str(i + 1) + ']/div/ul/li[2]/img'
        browser.find_element_by_xpath(seltd_lec).click()
        time.sleep(2)
        browser.find_element_by_xpath('/html/body').click()
        time.sleep(2)
        browser.switch_to.frame(browser.find_element_by_xpath('//iframe[@id="contentViewer"]'))
        tim = browser.find_element_by_xpath('/html/body/div[2]/div/div[4]/div[4]/span[2]')
        tim = tim.text
        t = tim_to_sec(tim)
        browser.switch_to.default_content()
        time.sleep(t)
        noflec = len(browser.find_elements_by_xpath('//*[@id="naviViewer"]/div[2]/ul/li'))
        for j in range(noflec - 1):
            selseltd_lec = '//*[@id="naviViewer"]/div[2]/ul/li[' + str(j + 2) + ']/div[2]'
            browser.find_element_by_xpath(selseltd_lec).click()
            time.sleep(2)
            browser.find_element_by_xpath('/html/body').click()
            time.sleep(2)
            browser.switch_to.frame(browser.find_element_by_xpath('//iframe[@id="contentViewer"]'))
            tim = browser.find_element_by_xpath('/html/body/div[2]/div/div[4]/div[4]/span[2]')
            tim = tim.text
            t = tim_to_sec(tim)
            browser.switch_to.default_content()
            time.sleep(t)




browser = webdriver.Chrome("/Users/zerohertz/Downloads/chromedriver")  #Directory지정
browser.implicitly_wait(5)  #암묵적으로 3초 딜레이
url = 'http://ecampus.konkuk.ac.kr'
browser.get(url)

browser.find_element_by_xpath('//*[@id="header"]/div[4]/ul/a/li').click()

id = input('ID : ')
pwd = input('Password : ')

browser.find_element_by_xpath('//*[@id="usr_id"]').send_keys(id)
browser.find_element_by_xpath('//*[@id="usr_pwd"]').send_keys(pwd)

browser.find_element_by_xpath('//*[@id="login_btn"]').click()

time.sleep(2)

NumOfReser = 0

while(1):
    mode = input('모드 선택!\n\n1. 예약 모드\t 2. 일반 모드\n\n1 또는 2를 입력하세요! : ')
    mode = int(mode)
    if mode == 1:
        end = 0
        No = []
        week = []
        while(end == 0):
            print_of_lec()
            a = input(str(NumOfReser + 1) + '번째 예약 과목 번호 : ')
            No = No + [int(a)]
            b = input(str(NumOfReser + 1) + '번째 예약 주차 : ')
            week = week + [int(b)]
            end = input('예약을 계속하시려면 0을 입력해 주세요(아니면 아무거나 누르세요) : ')
            end = int(end)
            NumOfReser = NumOfReser + 1
        for reser in range(NumOfReser):
            Ori = start_lec(No[reser], week[reser])
            play_lec(Ori)
            browser.get(url)
        print('\n\n수강 완료, 다음 강의를 선택하세용 ㅋㅋ\n\n\n')
        NumOfReser = 0
    elif mode == 2:
        print_of_lec()
        No = input('수업 번호 입력 : ')
        week = input('주차(Week) 입력 : ')
        No = int(No)
        week = int(week)
        Ori = start_lec(No, week)
        play_lec(Ori)
        browser.get(url)
        print('\n\n수강 완료, 다음 강의를 선택하세용 ㅋㅋ\n\n\n')
    else:
        print('\n\nError!\n\n\n')
~~~