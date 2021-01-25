# UBS-Comp

***DISCLAIMER: SCRAPING OF YELP CODE IS NOT INCLUDED IN THE REPOSITORY AS IT IS AGAINST YELP TO SCRAPE THEIR DATA***

# Contents
[Abstract](https://github.com/aakula7/UBS-Comp/blob/master/README.md#abstract)

[Project Proposal](https://github.com/aakula7/UBS-Comp/blob/master/README.md#project-proposal)
[Overview](https://github.com/aakula7/UBS-Comp/blob/master/README.md#overview)
[Goal](https://github.com/aakula7/UBS-Comp/blob/master/README.md#goal)
[Specifications](https://github.com/aakula7/UBS-Comp/blob/master/README.md#specifications)
[Data](https://github.com/aakula7/UBS-Comp/blob/master/README.md#data)
[Application Backend Structure](https://github.com/aakula7/UBS-Comp/blob/master/README.md#application-backend-structure)
[References](https://github.com/aakula7/UBS-Comp/blob/master/README.md#references)

# Abstract
Currently small businesses account for 99.9 percent of all U.S. business, however half of all small businesses fail within their first year, and more than 95 percent fail within their first five years. Restaurants make up a good portion of small business and are the hardest hit during recessions. Due to COVID, the restaurant industry has lost a net of 417,000 jobs and over $50 Billion in sales as of March and April, respectively. Algorithms cannot always predict when the next recession will hit, however we should be able to predict the future of small businesses and how to help them. We will develop an early warning system to better understand the trajectory of a small business, using restaurants across the state of New Jersey. The model takes into account features such as restaurant details, marketing and specials, restaurant safety and demographics. We are applying concepts of neural networks, LSTM, and K-Means Clustering  to predict if the restaurant will fail or survive over time.

# Project Proposal

## Overview
Develop an early warning system for small businesses in New Jersey which are at risk of closure using current and historic economic data. With the help of deep learning LSTM, neural networks, and K-Means Clustering, if we can predict the restaurants that will not survive then we will have a better understanding of the restaurant market and will be able to help clients struggling as well as those striving to prosper.

## Goal
  1.	Predict restaurants which will survive in the state of New Jersey
      a.	Combine data regarding time-series revenue, with demographics, reviews, safety, and other restaurant details
      b.	Build deep learning model to predict future sales and classify the restaurant as survivor or failure

## Specifications

### Reason
Restaurants are vital for the local culture and economy, as well as most times the only means of income for the family owning the restaurant. They make a huge difference in the economy as they can either bring in tons of new jobs for the community. (Bartik, 2020) However, as COVID-19 rages across the nation sending us into another recession, the question remains on how the restaurant industry will recover, particularly small, non-chain restaurants. Since March there have been mass layoffs and a risk of closure, which has been negatively associated with the expected length of the crisis. (Dixon, 2020) We choose this problem statement in hopes of building an early warning application, which can be deployed to help small businesses owners through these tough times and keep their dreams alive.

## Data
***Restaurant Details***
***Revenue***
Revenue management rests on a measure of time involved in the guest-service cycle. However, such time-series measurements raise complications for restaurants, as they explicitly sell meals, rather than a period of time at the table. An effective restaurant time series analysis calculation is revenue per available seat-hour (RevPASH), which is calculated through the use of total revenue for a given period divided by the product of the number of available seats and the length of the period in question. (Kimes, 1999).

***Alcoholic Beverage Control Licence***
Studies have shown that alcoholic beverages have helped restaurant sales increase by 69 to 87 percent. Overall giving a significant increase in sales due to alcoholic sales. (Wansink, 2006) Classifying each restaurant as a provider of alcohol and happy hour sales can help the neural network decide the future of the small business.

***MWBE***
Minority owned businesses are more prone to failing within the first 7 years of operation, primarily due to low access to financial resources, business knowledge, and international customers. (Le, 2015) Factors that would be helpful in analyzing the prosperity of minority owned businesses are hard work, interest of passion, family support, location, and food and service quality. 
           
***Delivery Service***
Changes in consumer demographic composition, information technology, labor force participation, and time have created demand as well as opportunities for delivering food using alternative methods. Studies show that younger consumers, individuals with at least college education, and households having a larger food budget are more likely to use these services. Gender, employment, marital status, and driving distance all also have an effect on interest in using delivery services. (Hossain, 1970)

***Outdoor Dining***
Restaurants at times have more requests for outdoor tables than it can fulfill, some diner even bypassing available dining room and waiting north of 30 mins for an outdoor seat. By leveraging skyline views, waterfront vistas, sunsets, people watching, and more, outdoor dining can offer a sense of time, place and wonder that even the most intriguing indoor spaces cannot. Outdoor dining can also add value to a restaurant by increasing the seating capacity and boosting revenue potential. (Smith, 2014)

***Popup/Food Truck***
As COVID-19 restrictions continue to spread in hopes of flattening the cure, food trucks have been considered as essential businesses. Most food trucks also have catering side hustles, and most also post up at local breweries on weekday nights or weekends for special events and pop-ups. With these events being canceled and stay at home orders ongoing, it is becoming much harder for food trucks to prosper. (Doyle, 2020)

***Dinner Reservations***
Small businesses owners in the restaurant industry might have a limited number of tables, therefore making it difficult to manage a large number of people interested in dining. Customers waiting for at least 10-20 minutes can increase their frustration and a possibility of losing customers. (Rarh, 2018) Opentable data shows the impact of dining reservations and the impact COVID is having on the industry, which will be valuable for present-day analysis. (State of the Industry, 2020)

**Marketing/Specials**
Companies need to be concerned with the future revenue and profit streams associated with the ongoing satisfaction and retention of their core, profitable customer bases. The companies that fail to recognize this truth do not survive nor prosper. (The Ultimate Guide to Customer Success, 2020)

***Social Media Activity/Restaurant Reviews***
Consumers are increasingly relying on social media to learn about unfamiliar brands. (Hanaysha, 2016) Social media advertisements have a significant positive effect on all dimensions of brand equity in the food industry. It is even more important to advertise through social media during the time of COVID. One method of interacting with customers are through online reviews, which create cultural and financial value for individual restaurants and also construct a positive or negative image of their locations that may lead to economic investment. (Zukin, 2015)

***Loyalty Programs***
Loyal customers have a greater lifetime value, as they generate 60 to 70 percent more revenue. (Mattila, 2001) Loyal customers also like to hear from the restaurant as 65 percent of them want stores they frequently visit to email them coupons, sales, and promotions. The most successful small businesses get over 60 to 70 percent of their customers coming back on a monthly basis. (Keh, 2006)

**Restaurant Safety**
If customers perceive a threat to be severe or if they have a belief that responding in a certain way will reduce the risk of a food safety concern, they are more likely to intervene and eliminate the threat. (Harris, 2020) Safety has become more important during the age of COVID, which is driving the change in consumer behavior.

**Demographics**
Income, age, household size and income, urbanization, day of the week, and season of the year all affect the choice of a particular foodservice. The demand for evening meals at full-service restaurants is likely to increase due to aging of “baby boomers”, increasing household income, and decreasing household size. On the other hand, quick-service restaurants should consider focusing on the needs of households with young children. (Kim, 2008)


## Application Backend Structure

![Application Backend Structure](https://github.com/aakula7/UBS-Comp/blob/master/Application%20Backend%20Structure.PNG)

**Application Backend Structure Description**
The restaurant reviews and information are stored in a database and accessible by wealth manager employees, restaurant owners, and other bank employees. By entering a zipcode they are able to extract the information they need from the database, and if a zipcode data is not available, then the bank employees can request the application to scrape the information from Yelp. If the information for that zipcode was present, then the data is clustered according to affordability. As much of the data necessary for this project is confidential, we were not able to get access to sales, therefore we used the safety ratings and a paper that analyzed revenue generation according to safety rating in the New York City area, to methematically calculate the possible revenue generated for the restaurants. The restaurant of interest and the generated sales are then fed into the models Arima Forecasting, LSTM Anomaly Detection, and LSTM Multivariate modeling to develop a greater understanding of the restaurant its survivability.

## References

Bartik, A., Bertrand, M., Cullen, Z., Glaeser, E., Luca, M., & Stanton, C. (2020, July 28). The impact of COVID-19 on small business outcomes and expectations. Retrieved from https://www.pnas.org/content/117/30/17656

Wansink, B., G. (2006, November 01). Wine Promotions in Restaurants: Do Beverage Sales Contribute or Cannibalize? - Brian Wansink, Glenn Cordua, Ed Blair, Collin Payne, Stephanie Geiger, 2006. Retrieved from https://journals.sagepub.com/doi/abs/10.1177/0010880406294656

Dixon, V. (2020, March 24). By the Numbers: COVID-19's Devastating Effect on the Restaurant Industry. Retrieved from https://www.eater.com/2020/3/24/21184301/restaurant-industry-data-impact-covid-19-coronavirus

Doyle, T. (2020, April 09). Boston Food Trucks Are Struggling to Cope With the COVID-19 Pandemic. Retrieved from https://boston.eater.com/2020/4/9/21213493/boston-food-trucks-struggling-coronavirus-pandemic

Hanaysha, J. (2016, April). The Importance of Social Media Advertisements in Enhancing Brand Equity: A Study on Fast Food Restaurant Industry in Malaysia. Retrieved from http://www.ijimt.org/vol7/643-M738.pdf

Harris, K., Depietro, R. B., Klein, J., & Jin, D. (2020, May 29). The Impact of Social Norms and Risk Assessment on Diners' Reaction to Food Safety Concerns in Restaurants. Retrieved from https://doi.org/10.1080/15378020.2020.1765947

Hossain, F., & Adelaja, A. (1970, January 01). Consumers’ Interest in Alternative Food Delivery Systems: Results From a Consumer Survey in New Jersey. Retrieved from https://ageconsearch.umn.edu/record/27754/

Keh, H., & Lee, Y. (2006, April 18). Do reward programs build loyalty for services?: The moderating effect of satisfaction on type and timing of rewards. Retrieved from https://www.sciencedirect.com/science/article/abs/pii/S0022435906000194

Kim, E., & Geistfeld, L. V. (2008, September 08). Consumers' Restaurant Choice Behavior and the Impact of Socio-Economic and Demographic Factors. Retrieved from https://doi.org/10.1300/J369v06n01_02

Kimes, S. (1999, June 01). Implementing Restaurant Revenue Management: A Five-step Approach - Sheryl E. Kimes, 1999. Retrieved from https://journals.sagepub.com/doi/abs/10.1177/001088049904000315

Le, N. (2015, December). Small Ethnic Restaurant Business Owner Strategies to Remain Operable Beyond 7 Years. Retrieved from https://scholarworks.waldenu.edu/dissertations/1903/

Mattila, A. S. (2001, December 01). Emotional Bonding and Restaurant Loyalty. Retrieved from https://journals.sagepub.com/doi/abs/10.1177/0010880401426009?journalCode=cqxa

Needles, A., & Thompson, G. M. (2013, February 05). Social Media Use in the Restaurant Industry: A Work in Progress. Retrieved from https://scholarship.sha.cornell.edu/cgi/viewcontent.cgi?referer=https://scholar.google.com/&httpsredir=1&article=1100&context=chrpubs

Rarh, F., Pojee, D., Zulphekari, S., & Shah, V. (2018, March 22). Restaurant table reservation using time-series prediction. Retrieved from https://ieeexplore.ieee.org/abstract/document/8321254

Smith, D. P. (2014, March 01). An Inside Look at Outdoor Dining. Retrieved from https://www.fsrmagazine.com/inside-look-outdoor-dining
State of the Industry. (2020). Retrieved from https://www.opentable.com/state-of-industry

Stutzman, W. (2020, July 20). Why pop-up shops, vending, outdoor dining, food trucks, and pick-up services are surging. Retrieved from https://www.foodtruckoperator.com/blogs/pop-up-shops-vending-outdoor-dining-food-trucks-and-pick-up-services-surge/

The Ultimate Guide to Customer Success. (2020). Retrieved from https://hi.fivestars.com/rs/187-VFC-057/images/Fivestars_UltimateLoyaltyGuide.pdf

Zukin, S., Lindeman, S., & Hurson, L. (2015, October 04). The omnivore's neighborhood? Online restaurant reviews, race, and gentrification. Retrieved from https://journals.sagepub.com/doi/abs/10.1177/1469540515611203
