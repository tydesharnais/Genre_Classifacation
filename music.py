from bs4 import BeautifulSoup
import requests

def refine(text: str):
    return text.replace('\n', '')
url = "https://www.billboard.com/charts/hot-100/"

result = requests.get(url)
doc = BeautifulSoup(result.text, 'html.parser')

ul = doc.find('ul', class_="lrv-a-unstyle-list lrv-u-flex lrv-u-height-100p lrv-u-flex-direction-column@mobile-max")
artist_place_peak = ul.find_all('span', class_="c-label")
song_Name = ul.find('h3', class_="c-title a-no-trucate a-font-primary-bold-s u-letter-spacing-0021 u-font-size-23@tablet lrv-u-font-size-16 u-line-height-125 u-line-height-normal@mobile-max a-truncate-ellipsis u-max-width-245 u-max-width-230@tablet-only u-letter-spacing-0028@tablet").text.replace('\t','')
artist = artist_place_peak[0].text.replace('\t', '')
peak = artist_place_peak[2].text.replace('\t', '')
wks_Chart = artist_place_peak[3].text.replace('\t', '')


print(f'{refine(song_Name)} {refine(artist)} {refine(peak)} {refine(wks_Chart)}')

# for div in divs:
#     ul = div.find('ul', class_="o-chart-results-list-row").ul
    
    #songName = ul.find('h3', class_="c-title a-no-trucate a-font-primary-bold-s u-letter-spacing-0021 lrv-u-font-size-18@tablet lrv-u-font-size-16 u-line-height-125 u-line-height-normal@mobile-max a-truncate-ellipsis u-max-width-330 u-max-width-230@tablet-only")
    #print(songName)
    #print(ul)
# li = doc.find_all('li', class_="o-chart-results-list__item")
# i =0
# for tags in li:
#     print(li.find("span"))
