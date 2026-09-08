import os
from jose import jwt
from dotenv import load_dotenv

load_dotenv()

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzg4ODY1NDM1LCJpYXQiOjE3ODg4NTY0MzUsImp0aSI6IjFhOWZjNjY5NjQyZjQ4MGI5ZjkwMTE1ZTEyMjU5MGQ4IiwidXNlcl9pZCI6IjU1In0.TibySo8fjNZ6Ki6Dcc3r7hMC45NOE27Q4Oc1nuQO7Hk"
secret = os.getenv("JWT_SECRET_KEY")

decoded = jwt.decode(token, secret, algorithms=["HS256"])
print(decoded)
