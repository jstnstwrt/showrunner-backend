import scrapy

class PlaceholderSpider(scrapy.Spider):
    name = 'placeholder'

    def start_requests(self):
        return []

    def parse(self, response):
        pass