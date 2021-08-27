"""Organized data for a single Reuters document."""
import re


class ReutersDocument():
    
    def __init__(self, id):
        self.id = id
        self.datetime = None
        self.topics = []
        self.places = []
        self.people = []
        self.orgs = []
        self.exchanges = []
        self.companies = []
        self.title = ""
        self.dateline = ""
        self.body = ""

    def add_datetime(self, datetime: str):
        self.datetime = datetime

    def add_topic(self, topic: str):
        self.topics.append(topic)

    def add_place(self, place: str):
        self.places.append(place)
        
    def add_person(self, person: str):
        self.people.append(person)
        
    def add_org(self, org: str):
        self.orgs.append(org)
        
    def add_exchange(self, exchange: str):
        self.exchanges.append(exchange)
        
    def add_company(self, company: str):
        self.companies.append(company)
    
    def add_title(self, title: str):
        self.title = title
    
    def add_dateline(self, dateline: str):
        self.dateline = dateline

    def add_body(self, body: str):
        self.body += body

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "datetime": self.datetime,
            "topics": self.topics,
            "places": self.places,
            "people": self.people,
            "orgs": self.orgs,
            "exchanges": self.exchanges,
            "companies": self.companies,
            "title": self.title,
            "dateline": self.dateline,
            "body": re.sub(r"\s+", r" ", self.body),   
        }
