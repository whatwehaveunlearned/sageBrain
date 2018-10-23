import scrapy
import json
from w3lib.url import add_or_replace_parameter
class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://www.kaggle.com/datasets.json?sortBy=hottest&group=all&page=1']

    def parse(self, response):
        data = json.loads(response.body) 
        total_results = data['totalDatasetListItems']
        page = 1
        # figure out how many pages are there and loop through them.
        for i in range(20, total_results, 20):  # step 20 since we have 20 results per page
            url = add_or_replace_parameter(response.url, 'page', page)
            yield scrapy.Request(url, self.parse_page)

        # don't forget to parse first page as well!
        yield from self.parse_page(self, response)

    def parse_page(self, response):
        data = json.loads(response.body) 
        # parse page data here
        for item in data['datasetListItems']:
            item = dict()
            pdb.set_trace()
            yield item
