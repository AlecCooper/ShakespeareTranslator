import pycurl
from io import BytesIO

b_obj = BytesIO()
crl = pycurl.Curl()

# Holds the headers from the curl request in check_404
headers = []

# Given a url and an the name of an output file, downloads the url
# to that file
def download_file(url, output_file_name):

    # Set URL value
    crl.setopt(crl.URL,url)

    # Write bytes that are utf-8 encoded
    crl.setopt(crl.WRITEDATA, b_obj)

    # Save the content to a text document
    file = open(output_file_name + ".txt", "wb")
    crl.setopt(crl.WRITEDATA, file)

    # Preform the file transfer
    crl.perform()

    # End curl session
    crl.close()

# given a URL checks to see if it gives a 404 error
def check_404(url):

    # clear headers list
    headers = []

    # Set URL value
    crl.setopt(crl.URL, url)

    # function to pass to crl.HEADERFUNCTION
    header_function = lambda header: headers.append(header)

    crl.setopt(crl.HEADERFUNCTION, header_function)

    # Preform the file transfer
    crl.perform()

    # End curl session
    crl.close()

    # Determine if 404 is in the header
    return "404" in headers[0].decode("utf-8").rstrip("\n")

# Read in filenames to a list
plays = []
play_file = open("plays.txt", "r")

for line in play_file:
    plays.append(line.rstrip("\n"))


# The base url
base_url = "https://www.litcharts.com/shakescleare/shakespeare-translations/"

# list of possible first url parts
acts = ["act","prologue/","epilogue/","induction/"]

