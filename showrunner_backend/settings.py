import os

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

BOT_NAME = 'showrunner_backend'
SPIDER_MODULES = ['showrunner_backend.spiders']
NEWSPIDER_MODULE = 'showrunner_backend.spiders'
