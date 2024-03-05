my_dict = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Accessing a value using a key
print("\nAccessing a value:")
print("Name:", my_dict["name"])

# Printing the keys and values
print("Keys:")
print(my_dict.keys())
print(my_dict.values())

# Adding a new key-value pair
my_dict["email"] = "aliceexample@gmail.com"
# Print dictionary
print("\nDictionary after adding a new item:")
print(my_dict)

# Removing an item
del my_dict["city"]

# Iterating over a dictionary
print("\nIterating:")
for key, value in my_dict.items():
    print(f"{key}: {value}")
