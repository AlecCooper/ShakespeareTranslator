from bs4 import BeautifulSoup
import os

# This method extracts a string from between html tags
def extract_string(tags):

    if tags.string != None:
        return tags.string
    else:
        print(tags)
        return "@"

# Given the location of a document, returns a 2d list of original and modern lines
def parse_document(document):

    # open document
    with open(document) as fp:
        soup = BeautifulSoup(fp,'html.parser')
    
    # Filter to just the rows comparing translation/original
    rows = soup.findAll("div",class_="comparison-row")

    # Filter out the title at the start of the comparison rows
    rows = rows[1:]

    # list of oirginal lines and translated lines
    original = []
    translation = []

    ctr = 0

    # Loop through translations
    for row in rows:

        # Seperate the original and translated sections
        original_content_col = row.div
        translated_content_col = row.div.next_sibling.next_sibling

        # Find the speaker text
        original_speaker_text = original_content_col.find("p", class_="speaker-text")
        translated_speaker_text = translated_content_col.find("p", class_="speaker-text")
        
        # Determine if both versions have a speaker text, if so we extract it
        if original_speaker_text != None and translated_speaker_text != None:
            
            # Extract the lines
            original_lines = original_speaker_text.findAll("span", class_=["line-mapping mapped","line-mapping"])
            translated_lines = translated_speaker_text.findAll("span",class_=["line-mapping mapped","line-mapping"])

            # We must concatenate all the lines with the same data id together
            # to get the full line
            if len(original_lines) > 0 and len(translated_lines) > 0:

                data_id = original_lines[0]["data-id"]
                # temp var to hold the strings we are concatenating
                line_string = extract_string(original_lines[0])
                
                # If there is more than one line, we need to loop through and extract them
                if len(original_lines) > 1:

                    # Loop through the remaining lines, concatenating identical ids
                    for line in original_lines[1:]:

                        # concate if same data id
                        if line["data-id"] == data_id:    
                            line_string += extract_string(line)      
                        else:
                            original.append((line_string,data_id))

                            # reset our holding variables
                            data_id = line["data-id"]       
                            line_string = ""

                else:       
                    original.append((line_string,data_id))

                # Preform the same procedure with the translated lines

                data_id = translated_lines[0]["data-id"]
                # temp var to hold the strings we are concatenating
                line_string = extract_string(translated_lines[0])
                
                # If there is more than one line, we need to loop through and extract them
                if len(translated_lines) > 1:

                    # Loop through the remaining lines, concatenating identical ids
                    for line in translated_lines[1:]:

                        # concate if same data id
                        if line["data-id"] == data_id:    
                            line_string += extract_string(line)    
                        else:
                            translation.append((line_string,data_id))

                            # reset our holding variables
                            data_id = line["data-id"]       
                            line_string = ""

                else:       
                    translation.append((line_string,data_id))
            
    return original, translation
    
# Lists holding the original, translated lines
original = []
translation = []

# Directory containing the webpages
#rootdir = os.getcwd() + "/www.litcharts.com/shakescleare/shakespeare-translations"

#test1, test2 = parse_document(rootdir + "/coriolanus/act-1-scene-1")
test1, test2 = parse_document("test.html")
print(test1)
print(test2) 

print(len(test1))
print(len(test2))

# We loop through all the directories and parse all the files
#for subdir, dirs, files in os.walk(rootdir):
 #   for file in files:
  #      print("Parsing" + os.path.join(subdir, file))
   #     new_original, new_translation = parse_document(os.path.join(subdir, file))

    #    print(new_original)

        #original = original.extend(new_original)
        #translation = translation.extend(new_translation)


        
# Save to text documents
#with open("original.txt", "w") as f:
 #   for line in original:
  #      f.write("%s\n" % line)

#with open("translation.txt", "w") as f:
 #   for line in translation:
  #      f.write("%s\n" % line)



        


