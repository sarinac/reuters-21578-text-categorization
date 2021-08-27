"""ReutersDocument factory: HTMLParser for SGML files for Reuters dataset."""
from reuters_parser.reuters_document import ReutersDocument
from html.parser import HTMLParser


class ReutersParser(HTMLParser):
    
    def __init__(self, encoding='latin-1'):
        HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding
        self.reuters_factory = []

    def _reset(self):
        """Reset all states (indicate if pointer is within a given tag)."""
        self.in_d = 0
        self.in_date = 0
        self.in_topics = 0
        self.in_places = 0
        self.in_people = 0
        self.in_orgs = 0
        self.in_companies = 0
        self.in_title = 0
        self.in_dateline = 0
        self.in_body = 0
        self.in_topic_d = 0
        self.doc = None

    def parse(self, feeed):
        """Decode and feed each line in the document to the parser."""
        for line in feeed:
            self.feed(line.decode(self.encoding))
        self.close()

    """
    Overwrite the defaults for HTMLParser handlers.
        * `handle_starttag` is called at the start of a tag.
        * `handle_endtag` is called at the end of a tag.
        * `handle_data` is called between the start and end tags.
    
    We will configure these handlers to call custom methods to 
    store different types of Reuters data.
    """
    
    def handle_starttag(self, tag, attrs):
        """Call the `start_TAG` method where TAG is the name of the tag.
        Do nothing if there is no corresponding method.
        """
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        """Call the `end_TAG` method where TAG is the name of the tag.
        Do nothing if there is no corresponding method.
        """
        method = 'end_' + tag
        getattr(self, method, lambda: None)()
    
    def handle_data(self, data):
        """Add data into the doc object based on its tag."""
        if self.in_d:
            if self.in_topics:
                self.doc.add_topic(data)
            elif self.in_places:
                self.doc.add_place(data)
            elif self.in_people:
                self.doc.add_person(data)
            elif self.in_orgs:
                self.doc.add_org(data)
            elif self.in_exchanges:
                self.doc.add_exchange(data)
            elif self.in_companies:
                self.doc.add_company(data)
        elif self.in_date:
            self.doc.add_datetime(data)
        elif self.in_title:
            self.doc.add_title(data)
        elif self.in_dateline:
            self.doc.add_dateline(data)
        elif self.in_body:
            self.doc.add_body(data)
        

    """
    These methods that are called upon the starttag and endtag handlers.
    Most of these start_ and end_ methods are to keep track of states.
    """
    def start_reuters(self, attributes):
        """Add a new doc object into the factory."""
        id = int(dict(attributes).get("newid", 0))
        self.doc = ReutersDocument(id)
        self.reuters_factory.append(self.doc)
    
    def end_reuters(self):
        self._reset()

    def start_date(self, attributes):
        self.in_date = 1

    def end_date(self):
        self.in_date = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0
        
    def start_places(self, attributes):
        self.in_places = 1

    def end_places(self):
        self.in_places = 0

    def start_people(self, attributes):
        self.in_people = 1

    def end_people(self):
        self.in_people = 0

    def start_orgs(self, attributes):
        self.in_orgs = 1

    def end_orgs(self):
        self.in_orgs = 0

    def start_exchanges(self, attributes):
        self.in_exchanges = 1

    def end_exchanges(self):
        self.in_exchanges = 0

    def start_companies(self, attributes):
        self.in_companies = 1

    def end_companies(self):
        self.in_companies = 0
        
    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_dateline(self, attributes):
        self.in_dateline = 1

    def end_dateline(self):
        self.in_dateline = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_d(self, attributes):
        self.in_d = 1

    def end_d(self):
        self.in_d = 0
