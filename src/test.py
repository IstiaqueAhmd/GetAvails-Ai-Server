import os
import jwt
from dotenv import load_dotenv

load_dotenv()

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiJ1cy0xIiwidG9rZW5fdHlwZSI6ImFjY2Vzc190b2tlbiIsImlhdCI6MTc3MDQzODAyMiwiZXhwIjoxNzcwNTI0NDIyfQ.6qqw0RpkkXQkpEhqTKSU9IvsJb5jL8YXUlFZFvJl8dE"
secret = os.getenv("JWT_SECRET_KEY")

decoded = jwt.decode(token, secret, algorithms=["HS256"])
print(decoded)
