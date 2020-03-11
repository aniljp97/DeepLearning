"""
Part 4:
BeautifulSoup fetching of course information
All results are printed to the output.
"""


import requests
from bs4 import BeautifulSoup

print("Fetching this course's information from the UMKC catalog using BeautifulSoup...")

html = requests.get("https://catalog.umkc.edu/course-offerings/graduate/comp-sci/")
soup = BeautifulSoup(html.content, "html.parser")

courses = soup.find_all(class_="courseblock")
c = soup.find(class_="courseblock")

print()

for course in courses:
    if course.span.text.split()[1] == "5590":
        print("\tCourse Name:\n", course.find(class_="title").text)
        print("\tCourse Overview:", course.find(class_="courseblockdesc").text)
