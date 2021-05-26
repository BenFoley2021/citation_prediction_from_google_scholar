# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:42:04 2021

@author: bcyk5
"""

import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains
from bs4 import BeautifulSoup
import codecs
import time
import pickle
import os

driver = webdriver.Chrome('./chromedriver')
def get_paper_titles2():
    """ Gets the paper titles from an html doc. Right now
        the html is loaded from a file.
    
    
    Returns
    -------
    papers : List
        Paper titles.

    """
    def get_papers_as_tags(page, papers):
        soup = BeautifulSoup(page, 'html.parser')
        for div in soup.find_all("a"):
            if "http://scholar.google.com/scholar_url?url" in str(div):
                papers.append(div)
        return papers
                
    path = os.getcwd() + "//test_gmail_app//updates_html//"
    papers = []
    for file in os.listdir(path):
        page = pickle.load(open(path + file, 'rb'))
        papers = get_papers_as_tags(page, papers)

    
    # for paper in papers:
    #     print(paper.text)
    
    return papers

def get_paper_titles():
    """ Gets the paper titles from an html doc. Right now
        the html is loaded from a file.
    
    
    Returns
    -------
    papers : List
        Paper titles.

    """

    
    
    #path = os.getcwd() + "//test_gmail_app//updates_html//"
    
    f = codecs.open("test_email_gs_update.html", 'r', "utf-8")
    page = f.read()
    
    soup = BeautifulSoup(page, 'html.parser')
    
    papers = []
    for div in soup.find_all("a"):
        if "http://scholar.google.com/scholar_url?url" in str(div):
            papers.append(div)
    
    # for paper in papers:
    #     print(paper.text)
    
    return papers

def find_paper_get_info(paper):
    
    
    """ Given a papers title, look it up.

    Parameters
    ----------
    paper : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    def find_paper_on_author_page():
        # finds and clicks the paper of interest
        paper_2_click = driver.find_elements_by_xpath(\
            "//*[text()='Impact of carbon-based charge transporting layer on the performance of perovskite solar cells']")
        
        paper_2_click = driver.find_element_by_xpath("//*[.='" + paper.text + "']")
        paper_2_click.click()
        return None
        
    def show_more():
        #author_info = getAuthorInfo(crawler)
        showMore = True
        
        while showMore == True:
            try:
                showMoreButton = driver.find_elements_by_xpath("//*[contains(text(), 'Show more')]") # getting the show more button
                
                parent = showMoreButton[0].find_element_by_xpath('..')
                parent2 = parent.find_element_by_xpath('..')
                
                if parent2.is_enabled() == True:
                    showMoreButton[0].click() # clicking
                    time.sleep(1)
                else:
                    showMore = False
                
                time.sleep(1)
                #print('clicked show more')
            except:
                print('couldnt click show more')
                showMore = False
           
        time.sleep(2)
    
    def search_for_paper():
        # constructs url and goes there
        
    # this doesn't work, need to get formula for url
    #https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Highly+Efficient+Wide-band-gap+Perovskite+Solar+Cells+Fabricated+by+Sequential+Deposition+Method%27&btnG=
    
        base_url = "https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q="
        end_url = "%27&btnG="
        paper_str = "+".join(paper.text.split(' '))
        
        url = base_url + paper_str + end_url
        
        driver.get(url)
        
        return None
        
    def get_authors_click_author():
        # gets the author ids, then clicks on the profile of the last one in
        # the list
        
        author_emails = driver.find_elements_by_class_name('gs_a')
        if author_emails:
            gs_id_links = author_emails[0].find_elements_by_css_selector('a')
        else:
            return 
        author_ids = set()
        author_link = None
        for author_link in gs_id_links:
            author_link_text = author_link.get_attribute('href')
            uInd = author_link_text.find("?user=") + len("?user=")
            endInd = author_link_text.find("&hl=")
                
            author_ids.add(author_link_text[uInd:endInd])
        
        return author_ids, author_link
        
    def get_data(paper):
            # actually gets data
                            # for the rest of the things, we want to limit our search the parent node which contains 
                # the data, instead of the whole page
        import re
        def get_data_field(soup, names):
            # takes the soup and figures out where the article was
            # published, and if it's listed as a jounral, confrence, or source
            def get_tag(soup, name):
                found = False
                tags = soup.findAll('div', text = re.compile(name))
                
                if len(tags) > 0:
                
                    for tag in tags:
                        if tag.text == name:
                            found = True
                            break
                    
                    if found == True:
                        parent = tags[0].find_parent('div')
                        kids = parent.findChildren()
                        
                        text_out = kids[1].text
                        return text_out
                    
                    else:
                        return None
                    
                else:
                    return None
            
            for name in names:
                result = get_tag(soup, name)
                if result:
                    return result, name
                
            return 'couldnt find', name
        
        
        tempDict = {}
        tempDict['titleID'] = paper.text
        html = driver.page_source
        soup = BeautifulSoup(html)
        # names = ["Pages"]
        # text_out, where_pub = get_where_pub(soup, names)
        try:
            tempDict['href url'] = driver.current_url
        except:
            tempDict['href url'] = 'couldnt get link'
            
        try:
            tempDict['Authors'] = get_data_field(soup, ['Authors'])[0]
        except:
            print('author exception')

        # need two steps to get these so the intermeidates are up in the function
        try:
            tempDict['pubDate'] = get_data_field(soup, ['Publication date'])[0]
        except:
            tempDict['pubDate'] = 'NA'
            print('pub date exepction')
        
        try:
            tempDict['all'] = driver.find_elements_by_id("gsc_ocd_bdy")[0].text
        except:
            tempDict['all'] = 'NA'
            print('all exception')
            
        names = ["Journal", "Source", "Conference", "Book"]
        try:
            text_data, name = get_data_field(soup, names) 
            tempDict[name] = text_data
        except:
            tempDict['journal'] = 'NA'
            print('journal exception')
            
        try:
            tempDict['vol'] = get_data_field(soup, ['Volume'])[0]
        except:
            tempDict['vol'] = 'NA'
            #print('volume exception')
            
        try:
            tempDict['issue'] = get_data_field(soup, ['Issue'])[0]
        except:
            tempDict['issue'] = 'NA'
            #print('issue exception')
            
        try:
            tempDict['pages'] = get_data_field(soup, ['Pages'])[0]
        except:
            tempDict['pages'] = 'NA'
            #print('pages exception')
        
        try:
            tempDict['publisher'] = get_data_field(soup, ['Publisher'])[0]
        except:
            tempDict['publisher'] = 'NA'
            print('publisher exception')
            
        try:
            tempDict['description'] = get_data_field(soup, ['Description'])[0]
        except:
            tempDict['description'] = 'NA'
            print('description excemption')

        try:
            cited = driver.find_elements_by_xpath("//div[@style='margin-bottom:1em']")    
            tempDict['cited'] = cited[0].text
        except:
            tempDict['cited'] = 'NA'
            print('cited execption')
            print(tempDict['titleID'])
                                
        try:
        # getting the graph of citation versus years
            years = driver.find_element_by_id("gsc_vcd_graph_bars").text
            citesYear = driver.find_elements_by_class_name("gsc_vcd_g_a")
            tempList = []
            for thing in citesYear:
                tempList.append(thing.find_element_by_css_selector("*").get_attribute('innerHTML'))
            tempDict['citedYear'] = {'year': years, 'cited': tempList}
        except:
            tempDict['citedYear'] = 'NA'
            print('couldnt get cited by year table')
    
        ######## getting the cited url element
        try:
            idEleParent = driver.find_elements_by_class_name("gsc_vcd_merged_snippet")
            idHref = idEleParent[0].find_element_by_class_name("gsc_oms_link").get_attribute('outerHTML')
            #tempDict['urlID'] = getCitedUrlID(idHref)
        except: 
            tempDict['urlID'] = "NA"
            
        # update the things regardless of how much data we actually got
        #paperDict.update({tempDict['titleID']: tempDict})
        ### will probably need to nest this in a try statement and have another error type for if it fails
        
        return tempDict
    
    search_for_paper()
    # driver now on the gs search result page for that paper
    
    author_ids, author_link = get_authors_click_author()
    if author_link:
        author_link.click()
    else:
        return None
    # driver now on the authors gs page

    show_more()
    # making sure all papers visible
    
    find_paper_on_author_page()
    # now on the page for that paper
    
    tempDict = get_data(paper)
    
    tempDict['Author_ids'] = author_ids
    
    return tempDict
    # 'Impact of carbon-based charge transporting layer on the performance of perovskite solar cells'
        
#authors = ?

def loop_papers_get_data():
    paperDict = {}
    papers = get_paper_titles2()
    paperDict = pickle.load(open('temp_paperDictA.pickle', 'rb'))
    for paper in papers:
        if paper.text not in paperDict:
            try:
                tempDict = find_paper_get_info(paper)
                #if tempDict:
                paperDict[paper.text] = tempDict
            except:
                paperDict[paper.text] = None
            
                #return paperDict
        
    return paperDict


if __name__ == "__main__":

    paperDict = loop_papers_get_data()
    
    path = os.getcwd() + '//paperDict files//'
    
    pickle.dump(paperDict, open("paperDictA_.pckl", 'wb'))
    
    
    
