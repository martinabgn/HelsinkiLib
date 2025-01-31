# Here is the script we need to finish for Week 2

#User input

print("Welcome to our search engine!")
print("You can use operators such as AND, OR, NOT in your query")
print("To exit press enter")
query = input("Please, write your query here: ")

try
    while query != "":


    #Here goes the code to process the query







#Handling incorrect input
except SyntaxError:
    print("Please, enter your query in the format of: word OPERATOR word separated by spaces")

except ValueError: #we need to raise this type of error in a word in not found in the database
    print("Unfortunately, we could not find", unknown, "in our data. Please, try other query")

except NOTError: #an error when NOT operator cannot produce the output as token is in every document :)
    print("Wow! Your query is found in EVERY document in our data!")

except:
    print("Sorry, something went wrong.")
