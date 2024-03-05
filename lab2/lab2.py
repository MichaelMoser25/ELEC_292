import requests
from bs4 import BeautifulSoup

http_text = requests.get('https://weather.com/en-CA/weather/tenday/l/ac1c001e07fc19e6a28d15a16800eb1a0136fc4c616009f0bfe15ebcee352be2').text

print(http_text)

soup = BeautifulSoup(http_text, 'lxml')

weather_data = soup.find_all('div', class_="DetailsSummary--DetailsSummary--1DqhO DetailsSummary--fadeOnOpen--KnNyF")

# Equal to number of lines, each of which contain weather for one day
### print(len(weather_data))

for day in weather_data:
    # DATE
    date = day.find('h3', class_="DetailsSummary--daypartName--kbngc").text
    # print(date)

    # TEMP
    temp_section = day.find('div', class_="DetailsSummary--temperature--1kVVp")
    span_tags = temp_section.find_all('span')
    # By using the ‘[0]’ index, we are telling python that we need the first <span>’s information
    max_temp = span_tags[0].text

    #Similarly, for scraping the minimum temperature type:
    min_temp = span_tags[2].span.text
    # print(max_temp)
    # print(min_temp)


    # WEATHER CONDITION
    weather_condition = day.find('div', class_="DetailsSummary--condition--2JmHb").span.text
    # print(weather_condition)


    # % CHANCE
    chance = day.find('div', class_="DetailsSummary--precip--1a98O").span.text
    # print(chance)


    # Wind Direction & Speed
    # wind_section = day.find('div', class_="DetailsSummary--wind--1tv7t DetailsSummary--extendedData--307Ax").span.text
    # wind_separated = wind_section.split()
    # print(wind_separated)
    wind = day.find('div', class_="DetailsSummary--wind--1tv7t DetailsSummary--extendedData--307Ax")
    span_dir = wind.find_all('span')
    wind_direction = span_dir[1].text
    wind_speed = span_dir[2].text

    final_data = (date, max_temp, min_temp, weather_condition, chance, wind_direction, wind_speed)
    print(final_data)

    with open('ELEC292_Lab2.txt', 'a') as f:
        print(final_data, file=f)


# Note that this command (‘with’ and ‘print’) were both placed in the main ‘for loop’ that
# we had created earlier. What, if any, is an efficiency with this code? How can we slightly change
# the code to make it more efficient and reduce some redundant actions? Explain this in your report,
# and write the code that addresses this question, and name it ‘ELEC292_Lab2_updated.py’.
#---------- Answer
# since in a for loop, process is executed from top to bottom then looped back to the top,
# this means that every time the loop gets to the end 'ELEC292_Lab2.txt' is opened and printed into the file
# The code can be modified to save all scrapped values into a dictionary and then writing to the file one singular time.
