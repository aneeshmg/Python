from bs4 import BeautifulSoup
import sys
import requests
import re

base_url = "https://www.imdb.com"
details_url = BeautifulSoup(requests.get(base_url + "/find?q=" + "+".join(sys.argv[1:])).text, "lxml").td.a['href']

search_res = BeautifulSoup(requests.get(base_url + details_url).text, "lxml")

name = search_res(itemprop="name")[0].get_text()
rating = search_res(itemprop="ratingValue")[0].get_text()
summary = search_res(class_="summary_text")[0].get_text().strip(' \t\n\r')

print(name + " -- " + rating)
print(summary)