import pandas as pd
import warnings
import psycopg2
import sys
from sqlalchemy import create_engine
import sqlalchemy
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import folium
import datetime
import calendar
from data_collect_utils import *

def extract(zipCodes = [], resName = None, host = 'localhost', user = None, word = None, attrLoadDir = None, reviewLoadDir = None, dohmhLoadDir = None, plot = False, latitude = 40.7393, longitude = -74.0020):
    """
    EXTRACT DATA TABLES FROM DATABASE FOR FURTHER ANALYSIS

    Inputs:
        :param zipCodes: Single or multiple zipcodes restaurant attributes and reviews to extract for analysis
        :param resName: Restauarant of interest for analysis
        :param host: Host connection for connecting to the database
        :param user: Username for connecting to database
        :param word: Password for connecting to database
        :param attrLoadDir: Directory to load attribute table
        :param reviewLoaddir: Directory to load review table
        :param dohmhLoadDir: Directory to load DOHMH table
        :param plot: if zipcodes should be plotted
        :param latitude: Latitude of city to plot
        :param longitude: Longitude of city to plot
    """

    resName = resName.lower()

    if user is None and word is None:
        warnings.warn('***** USERNAME AND/OR PASSWORD WAS NOT GIVEN *****')
        if attrLoadDir is None or reviewLoadDir is None or dohmhLoadDir is None:
            raise Exception('**** LOADING DIRECTORY WAS NOT PROVIDED FOR ATTRIBUTE, REVIEW, OR DOHMH DATA TABLE ****')
        else:
            print('**** LOADING DATA TABLES FROM PROVIDED DIRECTORIES ****')
            attrDF = pd.read_csv(attrLoadDir)
            reviewDF = pd.read_csv(reviewLoadDir)
            dohmhDF = pd.read_csv(dohmhLoadDir)

            return (attrDF, reviewDF, dohmhDF)
    else:
        try:
            conn_string =f"host={host} dbname='Restaurants' user={user} password={word}"
            conn = psycopg2.connect(conn_string)
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            myeng = create_engine(f"postgresql+psycopg2://{user}:{word}@{host}:5432/Restaurants")
            print('**** CONNECTED TO RESTAURANTS DATABASE SUCCESSFULLY ****')

            ## HELPER FUNCTIONS
            def checkDuplicates(myeng = myeng, cursor = cursor):
                print('**** CHECKING FOR DUPLICATES IN ATTRIBUTES AND REVIEWS DATA ****')
                resAttr_dupl = myeng.execute('select "Name","ZipCode", count(*) from "yelp"."attributes" group by ("Name","ZipCode") having count(*)>1').fetchall()

                if len(resAttr_dupl) == 0: 
                    print("**** THERE WERE NO DUPLICATES FOUND IN ATTRIBUTES TABLE ****") 
                else:
                    print("**** DUPLICATES WERE FOUND IN ATTRIBUTES TABLE AND ARE BEING DELETED ****")
                    delAttrDupl(myeng = myeng)  
                    
                resReviewDupl = myeng.execute('select  count(*),"Name","Date","Review" from "yelp"."reviews" group by ("Name","Date","Review") having count(*)>1').fetchall()        
                if len(resReviewDupl) == 0: 
                    print("**** THERE WERE NO DUPLICATES FOUND IN REVIEWS TABLE ****") 
                else:
                    print("**** DUPLICATES WERE FOUND IN REVIEWS TABLE AND ARE BEING DELETED ****")
                    delReviewDupl(myeng)
                        
            def delAttrDupl(myeng = myeng):
                myeng.execute('DELETE FROM "yelp"."attributes" WHERE "id" IN (SELECT "id" FROM (SELECT "id",ROW_NUMBER() OVER( PARTITION BY "Name","ZipCode" ORDER BY  "id" ) AS row_num FROM "yelp"."attributes" ) t WHERE t.row_num > 1 ) ')
                print("**** DUPLICATES DELETED FROM ATTRIBUTES TABLE ****")
                
            def delReviewDupl(myeng = myeng):
                myeng.execute('DELETE FROM "yelp"."reviews" WHERE "id" IN (SELECT "id" FROM (SELECT "id", ROW_NUMBER() OVER( PARTITION BY "Name","Date","Review" ORDER BY  "id" ) AS row_num FROM "yelp"."reviews" ) t WHERE t.row_num > 1 ) ' )
                print("**** DUPLICATES DELETED FROM REVIEWS TABLE ****")
            
            def extResDF(conn = conn, zipCodes = zipCodes):
                print('**** EXTRACTING ATTRIBUTE AND REVIEW DATA FOR ZIPCODES ****')

                zipCodes_new = [str(code) for code in zipCodes]
                zipCodes_str = ','.join(zipCodes_new)

                resAttrDF= pd.read_sql(f'select * from "yelp"."attributes" where "ZipCode" = ({zipCodes_str})', con = conn)                    
                resReviewDF = pd.read_sql(f'select * from "yelp"."reviews" where "ZipCode" = ({zipCodes_str})', con = conn)
                dohmhDF= pd.read_sql(f'select * from "Yelp"."DOHMH" where "ZipCode" = ({zipCodes_str})', con = conn)

                return (resAttrDF, resReviewDF, dohmhDF)

            ## CHECKING IF ZIPCODE EXISTS
            print('**** CHECKING IF ZIPCODE DATA EXISTS ****')
            zipCodes_new = [str(code) for code in zipCodes]
            zipCodes_str = ','.join(zipCodes_new)
            zipCodeData = myeng.execute(f'select * from "yelp"."zipcode" where "ZipCode" IN ({zipCodes_str})').fetchall()
            if len(zipCodeData) == 0: 
                raise Exception("**** ZIPCODE DATA WAS NOT FOUND ****")
            
            ## CHECKING IF RESTAURANT DATA EXISTS 
            else:
                print("**** ZIPCODE DATA WAS FOUND...CHECKING IF RESTAURANT DATA EXISTS ****")

                resAttrData = myeng.execute('select * from "yelp"."attributes" where "Attributes"."Name" = \''+resName+'\'').fetchall()
                resReviewData = myeng.execute('select * from "yelp"."reviews" where "Reviews"."Name" = \''+resName+'\'').fetchall()
                if len(resAttrData) == 0 or len(resReviewData) == 0: 
                    warnings.warn("**** RESTAURANT DATA WAS NOT FOUND ****")

                    ## CHECKING IF ZIPCODE EXISTS IN ATTRIBUTE AND REVIEW TABLES
                    print('**** CHECKING IF ZIPCODE DATA EXISTS IN RESTAURANT ATTRIBUTE AND REVIEW TABLES ****')
                    zipAttrData = myeng.execute(f'select * from "yelp"."zipcode" where "ZipCode" IN ({zipCodes_str})').fetchall()
                    zipReviewData = myeng.execute(f'select * from "yelp"."zipcode" where "ZipCode" IN ({zipCodes_str})').fetchall()
                    if len(zipAttrData) == 0 or len(zipReviewData) == 0:
                        warnings.warn('**** DATA DOES NOT EXIST FOR ZIPCODE IN ATTRIBUTE OR REVIEWS TABLE ****')

                        scraping = input('Would you like to scrape restaurant data for the zipcodes requested')
                        if scraping.lower() == 'yes' or scraping.lower() == 'y':
                            ## SCRAPING YELP DATA
                            print('**** SCRAPING YELP DATA ****')
                            warnings.warn('**** THIS WILL TAKE A COUPLE OF DAYS DEPENDING ON THE VOLUME OF DATA TO SCRAPE AND COMPUTER POWER ****')
                            scrapeYelpResAttr(PostalCodes = zipCodes, host = 'localhost', user = user, word = word, outputPath = None)
                            scrapeYelpResRev(PostalCodes = zipCodes, host = 'localhost', user = user, word = word, outputPath = None)

                            ## CHECKING FOR DUPLICATES AND EXTRACTING DATA
                            checkDuplicates(myeng = myeng, cursor = cursor)
                            resAttrDF, resReviewDF, dohmhDF = extResDF(conn = conn, zipCodes = zipCodes)

                            print('**** EXTRACTED RESTAURANT ATTRIBUTE, REVIEWS, AND DOHMH DATAFRAMES FROM DATABASE ****')
                            
                            if plot is True:
                                print('**** PLOTTING ZIPCODE DATA ON MAP ****')
                                zipCodeDF = gpd.GeoDataFrame.from_postgist(f'select * from "yelp"."zipcode"', con = conn, geom_col = 'geometry')
                                zipCodeDF['ZipCode'] = zipCodeDF['ZipCode'].astype(str)

                                m_1 = folium.Map(location = [latitude, longitude], tiles = 'cartodbpositron', zoom_start = 10)
                                folium.GeoJson(zipCodeDF, style_function = lambda feature:{'weight':1,'opacity':100}).add_to(m_1)
                                folium.GeoJson(zipCodeDF[zipCodeDF['ZipCode'].isin(zipCodes)], style_function = lambda feature:{'weight':3, 'fillcolor':'#228B22', 'color': 'red' }).add_to(m_1)
                                m_1

                            return (resAttrDF, resReviewDF, dohmhDF)

                        elif scraping.lower() == 'no' or scraping.lower() == 'n':
                            raise Exception('**** DATA DOES NOT EXIST....PROGRAM CANNOT CONTINUE ****')
                        else:
                            print('**** INPUT WAS NOT UNDERSTOOD...PLEASE ENTER YES, NO, Y, OR N ****')
                            scraping = input('Would you like to scrape restaurant data for the zipcodes requested')
                            if scraping.lower() == 'yes' or scraping.lower() == 'y':
                                print('**** SCRAPING YELP DATA ****')
                                warnings.warn('**** THIS WILL TAKE A COUPLE OF DAYS DEPENDING ON THE VOLUME OF DATA TO SCRAPE AND COMPUTER POWER ****')
                                scrapeYelpResAttr(PostalCodes = zipCodes, host = 'localhost', user = user, word = word, outputPath = None)
                                scrapeYelpResRev(PostalCodes = zipCodes, host = 'localhost', user = user, word = word, outputPath = None)

                                ## CHECKING FOR DUPLICATES AND EXTRACTING DATA
                                checkDuplicates(myeng = myeng, cursor = cursor)
                                resAttrDF, resReviewDF, dohmhDF = extResDF(conn = conn, zipCodes = zipCodes)

                                print('**** EXTRACTED RESTAURANT ATTRIBUTE, REVIEWS, AND DOHMH DATAFRAMES FROM DATABASE ****')
                                
                                if plot is True:
                                    print('**** PLOTTING ZIPCODE DATA ON MAP ****')
                                    zipCodeDF = gpd.GeoDataFrame.from_postgist(f'select * from "yelp"."zipcode"', con = conn, geom_col = 'geometry')
                                    zipCodeDF['ZipCode'] = zipCodeDF['ZipCode'].astype(str)

                                    m_1 = folium.Map(location = [latitude, longitude], tiles = 'cartodbpositron', zoom_start = 10)
                                    folium.GeoJson(zipCodeDF, style_function = lambda feature:{'weight':1,'opacity':100}).add_to(m_1)
                                    folium.GeoJson(zipCodeDF[zipCodeDF['ZipCode'].isin(zipCodes)], style_function = lambda feature:{'weight':3, 'fillcolor':'#228B22', 'color': 'red' }).add_to(m_1)
                                    m_1

                                return (resAttrDF, resReviewDF, dohmhDF)

                            elif scraping.lower() == 'no' or scraping.lower() == 'n':
                                raise Exception('**** DATA DOES NOT EXIST....PROGRAM CANNOT CONTINUE ****')
                            else:
                                raise Exception('**** INPUT WAS NOT UNDERSTOOD....PROGRAM CANNOT CONTINUE ****')
                    ## ELIF STATEMENT ##
                    else:
                        warnings.warn('**** ZIPCODE DATA EXISTS...BUT RESTAURANT DATA DOES NOT EXIST ****')
                        scraping = input('Would you like to scrape restaurant data for the zipcodes requested')
                        if scraping.lower() == 'yes' or scraping.lower() == 'y':
                            ## SCRAPING ZIPCODE DATA EVEN IF DATA EXISTS
                            print('**** SCRAPING YELP DATA ****')
                            warnings.warn('**** THIS WILL TAKE A COUPLE OF DAYS DEPENDING ON THE VOLUME OF DATA TO SCRAPE AND COMPUTER POWER ****')
                            scrapeYelpResAttr(PostalCodes = zipCodes, host = 'localhost', user = user, word = word, outputPath = None)
                            scrapeYelpResRev(PostalCodes = zipCodes, host = 'localhost', user = user, word = word, outputPath = None)
                            
                            ## CHECKING FOR DUPLICATES AND EXTRACTING DATA
                            checkDuplicates(myeng = myeng, cursor = cursor)
                            resAttrDF, resReviewDF, dohmhDF = extResDF(conn = conn, zipCodes = zipCodes)

                            print('**** EXTRACTED RESTAURANT ATTRIBUTE, REVIEWS, AND DOHMH DATAFRAMES FROM DATABASE ****')
                            
                            if plot is True:
                                print('**** PLOTTING ZIPCODE DATA ON MAP ****')
                                zipCodeDF = gpd.GeoDataFrame.from_postgist(f'select * from "yelp"."zipcode"', con = conn, geom_col = 'geometry')
                                zipCodeDF['ZipCode'] = zipCodeDF['ZipCode'].astype(str)

                                m_1 = folium.Map(location = [latitude, longitude], tiles = 'cartodbpositron', zoom_start = 10)
                                folium.GeoJson(zipCodeDF, style_function = lambda feature:{'weight':1,'opacity':100}).add_to(m_1)
                                folium.GeoJson(zipCodeDF[zipCodeDF['ZipCode'].isin(zipCodes)], style_function = lambda feature:{'weight':3, 'fillcolor':'#228B22', 'color': 'red' }).add_to(m_1)
                                m_1

                            return (resAttrDF, resReviewDF, dohmhDF)


                        elif scraping.lower() == 'no' or scraping.lower() == 'n':
                            ## CHECKING FOR DUPLICATES AND EXTRACTING DATA
                            print('**** EXTRACTING DATA FOR ZIPCODE WITHOUT RESTAURANT OF INTEREST ****')
                            checkDuplicates(myeng = myeng, cursor = cursor)
                            resAttrDF, resReviewDF, dohmhDF = extResDF(conn = conn, zipCodes = zipCodes)

                            print('**** EXTRACTED RESTAURANT ATTRIBUTE, REVIEWS, AND DOHMH DATAFRAMES FROM DATABASE ****')
                            
                            if plot is True:
                                print('**** PLOTTING ZIPCODE DATA ON MAP ****')
                                zipCodeDF = gpd.GeoDataFrame.from_postgist(f'select * from "yelp"."zipcode"', con = conn, geom_col = 'geometry')
                                zipCodeDF['ZipCode'] = zipCodeDF['ZipCode'].astype(str)

                                m_1 = folium.Map(location = [latitude, longitude], tiles = 'cartodbpositron', zoom_start = 10)
                                folium.GeoJson(zipCodeDF, style_function = lambda feature:{'weight':1,'opacity':100}).add_to(m_1)
                                folium.GeoJson(zipCodeDF[zipCodeDF['ZipCode'].isin(zipCodes)], style_function = lambda feature:{'weight':3, 'fillcolor':'#228B22', 'color': 'red' }).add_to(m_1)
                                m_1

                            return (resAttrDF, resReviewDF, dohmhDF)

                        else:
                            print('**** INPUT WAS NOT UNDERSTOOD...PLEASE ENTER YES, NO, Y, OR N ****')
                            scraping = input('Would you like to scrape restaurant data for the zipcodes requested')
                            if scraping.lower() == 'yes' or scraping.lower() == 'y':
                                print('**** SCRAPING YELP DATA ****')
                                warnings.warn('**** THIS WILL TAKE A COUPLE OF DAYS DEPENDING ON THE VOLUME OF DATA TO SCRAPE AND COMPUTER POWER ****')
                                scrapeYelpResAttr(PostalCodes = zipCodes, host = 'localhost', user = user, word = word, outputPath = None)
                                scrapeYelpResRev(PostalCodes = zipCodes, host = 'localhost', user = user, word = word, outputPath = None)
                            elif scraping.lower() == 'no' or scraping.lower() == 'n':
                                raise Exception('**** DATA DOES NOT EXIST....PROGRAM CANNOT CONTINUE ****')
                            else:
                                raise Exception('**** INPUT WAS NOT UNDERSTOOD....PROGRAM CANNOT CONTINUE ****')
                else:
                    print("**** RESTAURANT DATA WAS FOUND...EXTRACTING DATA ****")
                    
                    ## CHECKING FOR DUPLICATES AND EXTRACTING DATA
                    checkDuplicates(myeng = myeng, cursor = cursor)
                    resAttrDF, resReviewDF, dohmhDF = extResDF(conn = conn, zipCodes = zipCodes)

                    print('**** EXTRACTED RESTAURANT ATTRIBUTE, REVIEWS, AND DOHMH DATAFRAMES FROM DATABASE ****')
                    
                    if plot is True:
                        print('**** PLOTTING ZIPCODE DATA ON MAP ****')
                        zipCodeDF = gpd.GeoDataFrame.from_postgist(f'select * from "yelp"."zipcode"', con = conn, geom_col = 'geometry')
                        zipCodeDF['ZipCode'] = zipCodeDF['ZipCode'].astype(str)

                        m_1 = folium.Map(location = [latitude, longitude], tiles = 'cartodbpositron', zoom_start = 10)
                        folium.GeoJson(zipCodeDF, style_function = lambda feature:{'weight':1,'opacity':100}).add_to(m_1)
                        folium.GeoJson(zipCodeDF[zipCodeDF['ZipCode'].isin(zipCodes)], style_function = lambda feature:{'weight':3, 'fillcolor':'#228B22', 'color': 'red' }).add_to(m_1)
                        m_1

                    return (resAttrDF, resReviewDF, dohmhDF)


        except:
            raise Exception('**** CONNECTION TO RESTAURANTS DATABASE WAS UNSUCCESSFUL ****')