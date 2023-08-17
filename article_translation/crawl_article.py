# Import Library
import requests
from bs4 import BeautifulSoup
import re

def get_data(link):
    # scrap from website
    result = requests.get(link)
    wp = BeautifulSoup(result.text, "html.parser")

    full_text = []
    for paragraf in wp.find_all('p'):
        full_text.append(paragraf.string)

    full_text = [paragraf.strip() for paragraf in full_text if paragraf != None]
    full_text = ' '.join(full_text)
    full_text = re.sub(r"[-()\"#/@;<>`+=~|*]", "", full_text)
    full_text = full_text.split(".")

    return full_text
