from bs4 import BeautifulSoup

# Read in the xml file containing our links
link_file = open("data/links.xml","r")
contents = link_file.read()

# Create our soup
soup = BeautifulSoup(contents, "xml")

# List will hold our links for output
link_list = []

# Find every link in the file
links = soup.findAll("loc")

# Loop through links and add to our lists
for link in links:
    link_list.append(link.text)

# Write to file
with open("data/links.txt", "w") as f:
    for link in link_list:
        f.write("%s\n" % link)

