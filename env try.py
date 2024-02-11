import os
from dotenv import load_dotenv
load_dotenv()

a = os.getenv('EMAIL')
b = os.getenv('EMAIL_PASSWORD')
print(os.listdir())
print(a, b)
 