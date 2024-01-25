# Question 1
list_1 = [-1, 2, 3, 9, 0]
list_2 = [1, 2, 7, 10, 14]
print(list_1 + list_2)

my_list = [1, 2, 3, 4, 5]
for x in my_list:
    print(x)


empty_list = []
my_list = [10, 20, 30, 40, 50, 60]
for i in range(0, len(my_list)):
    print(my_list[i])
    empty_list.append(my_list[i])
print(empty_list)

# Question 2
list_1 = [-1, 2, 3, 9, 0]
list_2 = [1, 2, 7, 10, 14]
temp = []
for d in range(0, len(list_1)):
    temp.append(list_1[d]+list_2[d])
print(temp)


# Create a 1-D ndarry
import numpy as np
a = np.array([1, 2, 3, 4])

my_2d_array = np.array([[1, 2, 3],
                        [4, 5, 6]])

my_3d_array = np.array([[[1, 2], [3, 4]],
                        [[5, 6], [7, 8]]])

## By using ‘.shape’ property, we can figure out the shape of a given ndarray:

my_1d_array = np.array([1, 2, 3, 4, 5])
print(my_1d_array.shape)
# (5,) Note that for the 1-D array, the shape of our array is (5, ). This means there is only one axis in
# the array which has 5 elements as shown below:

my_2d_array = np.array([[1, 2, 3], [4, 5, 6]])
print(my_2d_array.shape)
# (2,3) For the 2D array, the shape is (2, 3), which means it is a 2x3 matrix

my_3d_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(my_3d_array.shape)
# (2,2,2) Finally, for the 3D array, the shape is (2, 2, 2), which means it is a 2x2x2 matrix

# Question #3: Create a 4D ndarray whose shape is (2, 3, 2, 1)
my_4d_array = np.array([[[[1], [2]], [[3], [4]], [[5], [6]]],
                    [[[7], [8]], [[9], [10]], [[11], [12]]]])
print(my_4d_array.shape)

# Here, we first create a 1D array which is filled with the numbers 0 to 29. Then using ‘.reshape()’,
# we change its shape to (5, 6). So, the result is a 2D array whose shape is (5, 6)
my_2d_array = np.arange(30).reshape(5, 6)
print(my_2d_array[1, 4]) #10
print(my_2d_array[2, :]) # Entire third row
print(my_2d_array[:, 2]) # Entire third column
print(my_2d_array[:, [1, 3]]) # Columns 2 and 3
print(my_2d_array[:, 1::2]) # Selects even number columns


# If we had a 100x100 matrix and we wanted to choose columns with indices 11, 13, 15, ..., 57, 59,
# we would use the following code:
print(my_2d_array[:, 11:60:2])
# Here, 11:60:2 indicates starting from index 11 and continuing to index 59 (60 is excluded) with a
# step size of 2
import numpy as np
# Question 4
my_array = np.array([[[0,  1,  2],
                        [3,  4,  5],
                        [6,  7,  8]],

                       [[9, 10, 11],
                        [12, 13, 14],
                        [15, 16, 17]],

                       [[18, 19, 20],
                        [21, 22, 23],
                        [24, 25, 26]]])
print(my_array[:, :, 0])
print()
print(my_array[1, 1, :])
print()
print(my_array[:, [0, 0, -1, -1], [0, -1, 0, -1]])

print()

# 10 12 31 14 -5 7 1        select 10 7 1
my_1_array = np.array([10,12,31,12,-5,7,1])
print(my_1_array[[0, 5, 6]])

print(my_2d_array[[0, 0, 2], [0, 5, 3]])

# Select every second element in a column
print(my_2d_array[0::2, [1, 3]])

print()

# Question 5
arr = np.arange(27).reshape((3, 3, 3))
print(arr)
print()
print(arr[(0, 1, 2), (1, 2, 0), (1, 2, 0)])
print(arr[1, (0, 2), (0, 2)])

#
#
# # Sum of columns
# sum_of_rows = my_ndarray.sum(axis=1)
#
# # [-45, -9, 27, 63, 99]
# indexing_array = sum_of_rows > 10
# # Prints [False, False, True, True, True]
#
# print(my_ndarray[indexing_array, :]) # returns all positive numbers

# Question 6
matrix = np.arange(-10, 20).reshape((5, 6))
print(matrix)
print()

values = matrix[:, np.sum(matrix, axis=0) % 10 == 0]
print(values)




