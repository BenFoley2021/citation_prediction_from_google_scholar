# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 17:36:34 2021

author_list_2

need to add the title of every paper from an author to that authors data, want to track authors by key in addition to name

5-3 adding a thing to check if we got a 404, need to mark that author as bad in the db

the 404 page has the text "\n\nError 404"

@author: Ben
"""

import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from bs4 import BeautifulSoup
import re
import sqlite3
import os
import time
from datetime import datetime
import pickle
import random
import traceback

from langdetect import detect, detect_langs

class pathHeadA():
    """ this is to make it easier for me to keep track of the url and where to go next
        contains all the info which needs to be passed around when searching the current paper
        and its connections. the object is returned to the mainCrawl function where the results 
        of crawling a paper are extracted, and then it passed back up to the crawlPaper function
    
        the object contains the url, the id of the citation page (I hope there's 1 unique
        Id for each papers cited by'). page is the current in pageination
        
        The methods are for modifying the url to go the next page or the next articles
        cited by
    """
    def __init__(self, dbA_path: str, dbB_path: str, main_loop_error: int, error_log: list):
        """ constructing  object 
        """
        now = datetime.now()
        currentTime = now.strftime("%d/%m/%Y %H:%M")
        currentTime = currentTime.replace("/","-")
        currentTime = currentTime.replace(":","-")
        self.driver = webdriver.Chrome('./chromedriver')
        self.scrappedByID = '2_8_1 ' + str(currentTime) # this ID will be used to keep track of version found which papers and authors. the pathID arg will always be the same for this run of the crawl
        self.urlBase = "https://scholar.google.com/citations?user="
        self.url = None
        self.urlEnd = "&hl=en"
        self.authorStack = None
        self.authorsDone = []
        self.curAuthorID = None
        self.papers_got = 0
        self.papers_attempted = 0
        self.paper_error = {"in_a_row": 0, "total": 0} # the number of times all attempts to get info from the papers failded (try / execempts)
        self.author_error = False # if True, there were too many errors scraping the author we can't flag them as done in the db
        self.error_log = error_log 
        self.main_loop_error = main_loop_error ### other errors to keep track of which will be dealt with in the mainLoop
        self.alreadySeen = 0 # counter for how many papers we've already seen
        self.yeild = 0 # number of papers sucessfully gotten from a page
        self.author_info = {} # stores info on an author
        self.done_with_author = {"params": {"non_en": 0, "no_pubDate": 0, "missing_fields": 0, "confrence_no_cites": 0, \
                                 "old_no_cites": 0, "no_title": 0, "book": 0, "not_real_paper": 0}, "to_stop": False} # this will probably be a dict. need keep track of things which will be used to
                                    # determine if we are currently mining useless data
        
        # dictionaries which contain info the authors and papers. these are the product of the program
        self.paperDict = {} 
        self.authorDict = {}
        self.papersThisAuthor = [] # list of papers which were scrapped from the current author.
                                    # in general, just a container for holder paper id which need to be 
                                    # added to the paperAuthor table of the database.
                                    # can call update_db_papers_whenever and reset to []
        
        # the visted attribute is a connection to the database where all pathHeads report 
        # where they've been. this allows to run in parrellel with other instances of 
        # the same program. I should probably have the value in visited by the pathHead id
        # then I can have different conditions for what's added to the stack depending on 
        # whether this pathHead visited it or another one did
        self.visited = sqlite3.connect(dbB_path)
        self.authorTable = sqlite3.connect(dbA_path)
        
        self.hit_capatcha = False
        
    def are_getting_useful(self, tempDict):
        """ there is often lots of junk at the bottom of authors gs pages, can't afford to waste time on it
            
            Junk is considered to be: works which weren't published in a journal, confrence, patent office'
            Confrence presentations which were never cited (there are lots of these, need to cut them off at some
            point), really old papers (removing these will be biasing the data, but often these were published before
            before citations were electronically track, so theirs won't be accurate anyway'), Non-english (lots of these)
            
            this function should accept self and the temp dict. if the current paper is decided to be junk, need to 
            keep track of it. after a certain amount of junk, move on from this author
                                       
            priorities:
                if isn't in english'
                    https://codethief.io/how-to-check-if-a-word-is-english-or-not-in-python/
                
                
                if X amount of meta data missing
                if is really old and has few cits
                if have had X confrence papers in a row with no cites
            
            
        """
        def check_if_done_w_author():
            
            thresh3 = ['non_en', 'no_pubDate', "no_title", "not_real_paper"]
            thresh5 = ['confrence_no_cites', "old_no_cites", "missing_fields", "book"]
            for key in thresh3:
                if self.done_with_author['params'][key] > 9:
                    self.done_with_author['to_stop'] = True
                    return None
        
            for key in thresh5:
                if self.done_with_author['params'][key] > 9:
                    self.done_with_author['to_stop'] = True
                    return None
        
            count = 0
            for key in self.done_with_author['params'].keys():
                count += self.done_with_author['params'][key]
            
            if count > 21:
                self.done_with_author['to_stop'] = True
                return None
        
        def is_conference():
            # need to check the Journal and Source to see if it's actually a confrence

            confrence_words = ['confrence', 'confrence', 'abstract', 'Abstract', 'abstracts', 'Abstracts', "bulletin"]
            keys_list = list(tempDict.keys())
            
            if tempDict['cited'] == 'NA': # This means it doesn't have any cites
            
                if "Confrence" in keys_list and (tempDict['cited'] == "NA" or tempDict['cited'] == "couldnt find"):
                    self.done_with_author["params"]["confrence_no_cites"] += 1
                    print('confrence')
                # this is terrible coding, go back and fix it
                elif 'Journal' in keys_list:
                    if any(thing in tempDict['Journal'] for thing in confrence_words):
                        self.done_with_author["params"]["confrence_no_cites"] += 1
                        print('confrence')
                        
                elif 'Source' in keys_list:
                    if any(thing in tempDict['Source'] for thing in confrence_words):
                        self.done_with_author["params"]["confrence_no_cites"] += 1
                        print('confrence')
                else:
                    self.done_with_author["params"]["confrence_no_cites"] = 0 # 
                    
            else: 
                self.done_with_author["params"]["confrence_no_cites"] = 0
                
            return None
            
        def is_book():
            keys_list = list(tempDict.keys())
            if "Book" in keys_list and (tempDict['cited'] == "NA" or tempDict['cited'] == "couldnt find"):
                self.done_with_author["params"]["book"] += 1
                print('book')
                
            for source in sources:
                if source in keys_list:
                    if re.search("book", tempDict[source], re.IGNORECASE):
                        self.done_with_author["params"]["book"] += 1
                        return None
                    
            else: # if we didn't find any evidence that it's a book reset the count
                self.done_with_author["params"]["book"] = 0 
                
        def is_old_no_cites():
            try:
                if int(tempDict['pubDate'].split("/")[0]) < 1990 and (tempDict['cited'] \
                        == "NA" or tempDict['cited'] == "couldnt find"):
                    self.done_with_author["params"]["old_no_cites"] += 1
                else:
                    self.done_with_author["params"]["old_no_cites"] = 0
            except:
                self.done_with_author["params"]["old_no_cites"] = 1
                
        def is_en():
                # this function also records if there isn't a title
            try:
                title = tempDict['titleID']
                if title != "NA" and  title != "couldnt find":
                    major_lang = detect_langs(title)[0]
                    self.done_with_author['params']["no_title"] = 0 # if there's a title then we reset this
                    
                    if major_lang.lang != 'en':
                        self.done_with_author['params']["non_en"] += 1
                    else:
                        self.done_with_author['params']["non_en"] = 0 
                    
                else:
                    self.done_with_author['params']["no_title"] += 1
            except:
                pass
            
        def is_pub_date():
            if tempDict['pubDate'] == "NA" or tempDict['pubDate'] == "couldnt find":
                self.done_with_author["params"]["no_pubDate"] += 1
            else:
                self.done_with_author["params"]["no_pubDate"] = 0 
                
        def is_missing_fields():
            missing_count = 0
            missing_thresh = 0.4
            for key in tempDict:
                if tempDict[key] == 'NA' or tempDict[key] == "couldnt find":
                    missing_count += 1
                        
            if missing_count / len(list(tempDict.keys())) > missing_thresh:
                self.done_with_author["params"]["missing_fields"] += 1
            else:
                self.done_with_author["params"]["missing_fields"] = 0 
                
        def is_real_paper():
            # people seem to put crap like their thesis, posters, papers that are still under review
            # or random other bs on their scholar. if we hit these want to make a note of it
            # for now this is kept as a seperate thing from confrences becuase it annoys me even more
            for source in sources:
                if source in tempDict:
                    for string in unwanted_sources:
                        if re.search(string, tempDict[source], re.IGNORECASE):
                            self.done_with_author['params']['not_real_paper'] += 1
                        else:
                            self.done_with_author['params']['not_real_paper'] += 0
                
        ########## start main function
        
        
        #start of main function script
        sources = ["Journal", "Source", "Conference", "Book"] # the possible places where the work could be published
        unwanted_sources = ["tbd", "under review", "in review", "not published", "poster", "bulletin", "workshop"]


        #is_conference() ### is it a confrences with no cites, getting confrences now
        is_book() # is it a book with no cites
        is_old_no_cites() # is the paper old with no cites
        is_en() # is the title in english
        is_pub_date() # is there a publication date
        is_missing_fields() # what fraction of feilds are missing
        is_real_paper()
        ######### checking if done with author
        check_if_done_w_author()
        
        return None
        
    def getAnAuthorList(self, limit):
        """ gets a list of authors to put in the stack and scrape. if fails, exit program
        
        Returns
        -------
        None.

        """
        
        sql = "SELECT id FROM " + 'perovskiteAuthors' + " WHERE scrapped=?" + " LIMIT " + limit 
        key = [("False")]
        authorStack = []
        with self.authorTable as con:
            data = con.execute(sql, key)
            for row in data:
                authorStack.append("".join(row))
        
        self.authorStack = authorStack
        return None
    
    def makeUrl(self):
        self.url = self.urlBase + self.curAuthorID + self.urlEnd
        return None
    
    def nextAuthor(self):
        """ move to the next author. if there aren't ant try and get more
            clears temporaru variables associted with collection of data from one author
            will probably also have this call to update the databases

        Returns
        -------
        None.

        """
        self.curAuthorID = self.authorStack.pop(0)
        
        if self.curAuthorID:
            self.authorsDone.append(self.curAuthorID)
        return None
    
    def update_db_authors(self):
        """ updates the authors table as to which have been scrapped. does this after scrapping a batch
            of authors
        

        Returns
        -------
        None.

        """
        sql = "UPDATE perovskiteAuthors" + " SET" + " scrapped=" + '1' + " WHERE id=?"
        
        for author in self.authorsDone:
            key = [(author)]
        
            try:
                with self.authorTable as con:
                    data = con.execute(sql, key)
            except:
                print('error setting author to scrapped')        
        
        self.authorsDone = [] #### reset authors after updating the db. 
        return None
    
    def update_db_one_author(self):
        """ updates the authors table with the current author id so we know we've done it
        
        Returns
        -------
        None.

        """
        sql = "UPDATE perovskiteAuthors" + " SET" + " scrapped=" + '1' + " WHERE id=?"
        
        try:
            with self.authorTable as con:
                data = con.execute(sql, [(self.curAuthorID)])
                print(self.curAuthorID)
        except:
            print('error setting author to scrapped')        
            
        return None
    
    def update_db_papers(self):
        """ updates the authorpapers table of the db_B with the title of the papers that were
            scapped in the last batch
        
        
        Returns
        -------
        None.
        
        """
        data = []
        for key in self.papersThisAuthor: # i suppose this loop structure could also be shared with 'papers' for more polymorphic-ness
            data.append((key, self.scrappedByID, self.curAuthorID))
        
        sql = 'INSERT INTO papers_from_authors_5_3 (id, value, author) values (?, ?, ?)'
    
        counter = 0
        for dataRow in data:
            try:
                with self.visited as con:
                    temp = con.execute(sql, dataRow)
                    counter += 1
                    #print('added paper to db')
            except sqlite3.IntegrityError:
                print('all good its already there')
            except:
                print('oh no an actual db error')
        print('sent ' + str(counter) + " to db")
        
        self.papersThisAuthor = [] # reseting
        return None
    
    def check_db_papers(self, id_to_check):
        """ Checks to see if a paper title is in the paperAuthor table
    
        Returns
        -------
        Bool.

        """
        
        sql = "SELECT EXISTS(SELECT 1 FROM " + "papers_from_authors_5_3" + " WHERE id=?)"

        #sql2 = 'SELECT EXISTS(SELECT 1 FROM TEST WHERE id=?)'
        key = [(id_to_check)]
        with self.visited as con:
            data = con.execute(sql, key)
            for row in data:
                if row[0] == 1:
                    return True
                else:
                    return False

    def getInfoFromPapers(self,papers):
        """ gets all necssary info and puts in a dict. like parse papers from paperGraphScrapper
        
        things we need: 
            paper ID
            title
            authors
            Publication date
            Journal
            Volume
            Issue
            Pages
            Publisher
            Description
            Total citations
            citation plot
    
        Parameters
        ----------
        papers : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        
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
        ########## end funct
        def reset_author_attrs():
            # reseting the data used to determine if we are getting useful papers
            self.done_with_author = {"params": {"non_en": 0, "no_pubDate": 0, "missing_fields": 0, "confrence_no_cites": 0, \
                             "old_no_cites": 0, "no_title": 0, "not_real_paper": 0, "book": 0}, "to_stop": False} # this will probably be a dict. need keep track of things which will be used to
                                # determine if we are currently mining useless data
            self.paper_error['in_a_row'] = 0 
            self.paper_error['total'] = 0 
            self.papers_got = 0
            self.papers_attempted = 0
            self.author_error = False
            self.papersThisAuthor = []
        def get_data():
            # actually gets data
                            # for the rest of the things, we want to limit our search the parent node which contains 
                # the data, instead of the whole page
            self.papers_got += 1 # as long as we can get title that means we at least loaded the page
            
            
            html = self.driver.page_source
            soup = BeautifulSoup(html)
            # names = ["Pages"]
            # text_out, where_pub = get_where_pub(soup, names)
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
                tempDict['all'] = self.driver.find_elements_by_id("gsc_ocd_bdy")[0].text
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
                cited = self.driver.find_elements_by_xpath("//div[@style='margin-bottom:1em']")    
                tempDict['cited'] = cited[0].text
            except:
                tempDict['cited'] = 'NA'
                print('cited execption')
                print(tempDict['titleID'])
                                    
            try:
            # getting the graph of citation versus years
                years = self.driver.find_element_by_id("gsc_vcd_graph_bars").text
                citesYear = self.driver.find_elements_by_class_name("gsc_vcd_g_a")
                tempList = []
                for thing in citesYear:
                    tempList.append(thing.find_element_by_css_selector("*").get_attribute('innerHTML'))
                tempDict['citedYear'] = {'year': years, 'cited': tempList}
            except:
                tempDict['citedYear'] = 'NA'
                print('couldnt get cited by year table')
        
            ######## getting the cited url element
            try:
                idEleParent = self.driver.find_elements_by_class_name("gsc_vcd_merged_snippet")
                idHref = idEleParent[0].find_element_by_class_name("gsc_oms_link").get_attribute('outerHTML')
                tempDict['urlID'] = getCitedUrlID(idHref)
            except: 
                tempDict['urlID'] = "NA"
                
            # update the things regardless of how much data we actually got
            tempDict['scrap_auth_id'] = self.curAuthorID
            self.paperDict.update({tempDict['titleID']: tempDict})
            

            ### will probably need to nest this in a try statement and have another error type for if it fails
            self.are_getting_useful(tempDict) ### checking usefullness of this paper
            
            return tempDict
            
        def decide_author_error():
            if self.paper_error['in_a_row'] > 3:
                self.author_error = True
                print('author error')

            elif self.papers_attempted > 5 and self.papers_got / self.papers_attempted < 0.8:
                self.author_error = True
                print('author error')
            return None
        
        ####### start of script within this function
        reset_author_attrs()     

        for i, paper in enumerate(papers):
            
            # need to get the title to check if we've seen it before
            # should be in paper gsc_a_at .text
            decide_author_error()
            if self.author_error == True:
                print('papers dont seem to be loading')
                return None
            
            titleID = paper.text 
            self.papersThisAuthor.append(titleID)
            if self.check_db_papers(titleID) == False: # this operates on the paper element, which we already have
                paper.click()
                #print('clicked')
                time.sleep(3)
                #print('clicked')
                tempDict = {}
                tempDict['titleID'] = titleID
                self.papers_attempted += 1
                try:
                    self.driver.find_element_by_id("gsc_vcd_title").text
                except:
                    print('first time trying to get gsc_vcd_title failed') ### somtimes the page takes awhile to load. if we can't find this node it means we need to wait longer
                    time.sleep(5)
                    
                try: # main actions of function occur here
                    tempDict['title_main'] = self.driver.find_element_by_id("gsc_vcd_title").text
                    self.paper_error['in_a_row'] = 0 
                    self.papers_got += 1
                    tempDict = get_data()
                    if self.done_with_author['to_stop'] == True: # if are_getting_useful method decided that we were
                                    # just scrapping junk, stop by returning None.
                                    # this will move to the next paper
                        print('scrapping junk moving to next author')
                        return None 
                    
                except: 
                    tempDict['title_main'] = 'NA'
                    print('title_main exception')
                    self.paper_error['in_a_row'] += 1 
                    self.paper_error['total'] += 1
                    
                ######## clicking the x button so the next paper will be visible, need to do this no matter what
                xButton = self.driver.find_element_by_id("gs_md_cita-d-x")
                xButton.click()
                time.sleep(1)
        return None

def getParentText(driver, textIn): # this function no longer used
    # gets the text of the parent node found from the textIn
    tempVar = driver.find_element_by_xpath("//*[contains(text(), '" + \
                                                   textIn + "')]")
    return tempVar.find_element_by_xpath('..').text 

def getCitedUrlID(textIn):
    """ getting the urlID by text parsing. the url ID is needed to continue the crawl, so
        if it's not in the right format we return none'

    """
    startInd = textIn.find('hl=en&amp;cites=') + len('hl=en&amp;cites=')
    textIn = textIn[startInd:]
    
    endInd = textIn.find('&amp')
    
    return isFloat(textIn[0:endInd])

def isFloat(numIn): # simple function to check if str can be converted to float
    try:
        float(numIn)
        return numIn
    except:
        #print('bad cited str')
        return None

def getAllPapers(crawler):
    
    time.sleep(2)
    author_info = getAuthorInfo(crawler)
    showMore = True
    
    while showMore == True:
        try:
            showMoreButton = crawler.driver.find_elements_by_xpath("//*[contains(text(), 'Show more')]") # getting the show more button
            time.sleep(1)
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
    
    papers = crawler.driver.find_elements_by_class_name("gsc_a_at") ##getting all the papers
    
    author_info['num_papers'] = len(papers)
    
    author_info = is_author_sketchy(papers,author_info)
    
    return papers, author_info

def is_author_sketchy(papers,author_info):
    import re # hooray I was supposed to get practice with re for tdi 
    ### ad hoc function to determine if the author is sketchy or isn't at an american / european ish insitution
    def is_author_usa(author_info):
        #checks to see if anything in the ver_emial or rank indicates non-us/eu ish instution
        for key in author_keys_to_check:
            if key in author_info:
                for string in non_usa_list:
                    if re.search(string, author_info[key], re.IGNORECASE):
                        author_info['usa'] = False
                        break
        return author_info
    
    indian_cities = ["India", "Indian", "Delhi", "Mumbai", "Kolkāta", "Kolkata", "Bangalore", "Chennai", \
                 "Hyderābād", "Hyderabad", \
                "Pune", "Ahmadabad", "Ahmadābād", "Sūrat", "Surat", \
                    "Lucknow", "Jaipur"]

    # chinese_cities = ["China" , "Chinese", "Shanghai", "Beijing", "Chongqing", "Tianjin", "Guangzhou", \
    #               "Shenzhen", "Chengdu", "Nanjing", "Wuhan"]
    other_places = ["indonesia", "china", "india", "malaysia", "africa", "Guatemala", "Nigeria", "brazil", \
                    "chile", "Brasília", "Brasilia", "mexico"]
        
    domains = [ "edu.in", "gov.in", "my.edu", "ac.in", "ac.id", "ac.my", "my.edu", "gov.br", \
               "ac.br", "edu.br", "edu.au", "gov.au", "ac.du"]
        #"gov.cn""edu.cn",
    non_usa_list = indian_cities + domains + other_places
    
    author_keys_to_check = ['verEmail', 'rank']
    author_info['usa'] = True
    
    author_info = is_author_usa(author_info)
    
    if len(papers) > 600:
        author_info['sketchy'] = True
        
    else:
        author_info['sketchy'] = False
    
    return author_info
    
def getAuthorInfo(crawler):
    
    tempDict = {}
    
    # 
    try:
        # this node has all the author info in its children, if we can't get it no need to try the rest
        authorEle = crawler.driver.find_elements_by_id("gsc_prf")[0]

        try:
            tempDict['name'] = authorEle.find_elements_by_id("gsc_prf_in")[0].text
        except:
            print('couldnt get author name')
        
        try:
            tempDict['rank'] = authorEle.find_elements_by_class_name("gsc_prf_ila")[0].text
        except:
            print('couldnt get author rank')
            
        try:
            tempDict['verEmail'] = authorEle.find_elements_by_id("gsc_prf_ivh")[0].text
        except:
            print('couldnt verify email')
        
        try:
            intEle = authorEle.find_elements_by_id("gsc_prf_int")[0]
            cats = intEle.find_elements_by_class_name("gsc_prf_inta")
            catsList = []
            for cat in cats:
                catsList.append(cat.text)
            tempDict['cats'] = catsList
        except:
            pass
            #print('couldnt get author cats')
                
    except:
        print('something went wrong getting author info')

    return tempDict

def saveCurrent(dict2Save, fileName):
    """ converting the current paperList to a dict and saving it
    """
    def savePickle(fileName, var2Save, outDir):
        fileName = fileName + '.pckl'
        path = outDir + '/' + fileName
        with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(var2Save, f)
        f.close()
        
    outDir = 'outputs'
    
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    dt = dt.replace('/','-')
    dt = dt.replace(':','_')
    
    fileName = fileName + '_' + dt    
    
    savePickle(fileName, dict2Save, outDir)

def capatcha(crawler):
    cap = crawler.driver.find_elements_by_css_selector("iframe[src^='https://www.google.com/recaptcha/api2/anchor?']") 
    if cap:
        crawler.hit_capatcha = True
        print(crawler.curAuthorID)
        input("Press Enter to continue...")
        crawler.hit_capatcha = False
    return None

def mainPaperScrapeLoop(dbA_path: str, dbB_path: str, main_loop_error: int, error_log: list) -> pathHeadA:
    """ need to 
    1) initialize object
    while contiune = True
    2) get authors to scrape
    3) scrape authors
    4) dump dict to disk
    5) update databases
    loop
    
    stop conditions:
        generic error
        run out of authors
        
        to add if have time: seen almost all the papers before

    Returns
    -------
    None.

    """
    def update_db_404():
        # need to set the author to 404
        sql = "UPDATE perovskiteAuthors" + " SET" + " scrapped=" + '404' + " WHERE id=?"
        key = [(crawler.curAuthorID)]
    
        try:
            with crawler.visited as con:
                data = con.execute(sql, key)
        except:
            print('error setting author to 404')    
                
    def is_404():
        html = crawler.driver.page_source
        soup = BeautifulSoup(html)
        test_str = "\n\nError 404"
        if test_str in soup.text:
            return True
        else:
            return False
        
    limit = str(1000) # how many authors to grab from the db (one author will be randomly selected from this list for crawl)
    paper_save_param = 1000 ### after this many papers, dump the paper data to disk
    crawler = pathHeadA(dbA_path, dbB_path, main_loop_error, error_log) # intialize object
    missed = []
    
    scrape = True
    try: # this try block is so the program saves any progress if manually interupted
        while scrape == True:
            pathHeadA.getAnAuthorList(crawler, limit) ### grabbing a bunch of authors
            crawler.authorStack = [random.choice(crawler.authorStack)] # picking only one to stay in the stack
            # side note: the stack isn't needed anymore, since the program has been changed to do
            # one author at a time. just haven't redone the object attrs yet
            crawler.driver.delete_all_cookies()
            pathHeadA.nextAuthor(crawler)
            pathHeadA.makeUrl(crawler)
            crawler.driver.get(crawler.url) # getting the url
            if is_404() == True:
                time.sleep(2)
                crawler.driver.get(crawler.url)
                if is_404() == True:
                    time.sleep(3)
                    crawler.driver.get(crawler.url)
                    if is_404() == True:
                        update_db_404()
                        crawler.author_error = '404'
                        continue
            
            #capatcha(crawler) # checking to see if there is a capatcha
            print(crawler.curAuthorID)
            papers, author_info = getAllPapers(crawler)

            if author_info['sketchy'] == False and author_info['usa'] == True: # if the author looks suspicious then we skip
                print('skipped author')
                if len(papers) > 0:
                    pathHeadA.getInfoFromPapers(crawler, papers)
                    crawler.author_info.update({crawler.curAuthorID: author_info})
                    crawler.author_info[crawler.curAuthorID]['all_papers'] = \
                       crawler.papersThisAuthor
                    print("len papers_this_author = " + str(len(crawler.papersThisAuthor)))
                    
                    
                    if crawler.author_error == False:
                        pathHeadA.update_db_one_author(crawler)
                        pathHeadA.update_db_papers(crawler)
                        print('marked author as done in db')
                    
                else:
                    missed.append(crawler.curAuthorID)     
                
                if len(crawler.paperDict) > paper_save_param:
                    saveCurrent(crawler.paperDict, 'paperDictA') ### dumping to disk
                    crawler.paperDict = {}  # reseting
                    saveCurrent(crawler.author_info, 'author_info') ### dumping to disk
                    crawler.author_info = {}
            
            else: # if the author was suspocious we record having been there
                pathHeadA.update_db_one_author(crawler)

    except:
        print('exception')
        print(traceback.format_exc())
        crawler.driver.quit()
        crawler.error_log.append(traceback.format_exc())
        saveCurrent(crawler.author_info, 'author_info') 
        saveCurrent(crawler.paperDict, 'paperDictA')
        crawler.main_loop_error += 1
        time.sleep(360 + random.randrange(1,100))
         # if we have an error, end the session. the webdriver will be reintiialized when mainloop called again
        #mainPaperScrapeLoop(dbA_path, dbB_path) # try to reset everything by calling itself
        return crawler
    return None

if __name__ == "__main__":
    
    dbA_path = 'C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\dataBase\\db_C.db'
    dbB_path = 'C:\\Users\\bcyk5\\OneDrive\\Documents\\ds projects big data\\get citations google scholar\\parallel scrape paper graph and author\\test1\\dataBase\\db_C.db'
    
    # the purpose of this loop is to restart the crawl if there's error that wasn't handled. Call 
    # mainPaperScrapeLoop again and get a new crawler object(with new webdriver and db connections)
    # the only thing thats carried over is the count of these errors. If it keeps throwing errors
    # stop trying
    scrape = True
    main_loop_error = 0
    error_log = []
    while scrape == True:
        crawler = mainPaperScrapeLoop(dbA_path, dbB_path, main_loop_error, error_log)
        saveCurrent(crawler.author_info, 'author_info') 
        saveCurrent(crawler.paperDict, 'paperDictA')
        main_loop_error = crawler.main_loop_error
        error_log = crawler.error_log
        if main_loop_error > 200:
            scrape = False


