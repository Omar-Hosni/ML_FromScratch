import html_text
import requests
from googletrans import Translator

translator = Translator()

page = requests.get('https://linuxhint.com/find-where-python-installed-windows/')

#print(html_text.extract_text(page.text))

allLines = str(html_text.extract_text(page.text))

print(translator.translate(allLines, dest="hi",src="en"))