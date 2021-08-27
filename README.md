# reuters-21578-text-categorization

The Reuters Corpus is a collection of documents with news articles which totals over ten thousand documents. The documents in the Reuters-21578 collection appeared on the Reuters newswire in 1987. It is widely used for text categorization research. The dataset can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection).

## Helpful Documents

* [How are the Reuters files organized?](https://link.springer.com/content/pdf/bbm%3A978-3-642-04533-2%2F1.pdf)

# Getting Started

All instructions were written for Mac OS, so try your best, Windows friends. 😏 😏 😏

## Setting up local environment

Clone the repo to your local machine.
```
git clone https://github.com/sarinac/reuters-21578-text-categorization.git
```

Install pipenv to manage your virtual environment.
```
brew install pipenv
```

Move to the current directory.
```
cd reuters-21578-text-categorization
```

Install python dependencies from `requirements.txt` into the virtual environment.
```
pipenv install -r requirements.txt
```

## Using virtual environment

Activate shell session.
```
pipenv shell
```

Deactivate using `CTRL+D`.

## Installing packages

Use `pipenv` to install or update packages.
```
pipenv install <package_name>
```

# Running Modules

There are currently 2 modules:
1. Downloader
2. Parser
