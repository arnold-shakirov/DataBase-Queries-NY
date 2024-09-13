import numpy as np
import rich
from rich import print

console = rich.get_console()

arr = np.array([[9,2,3], [0,2,4]])
arr2 = np.array([[-1,-2,0], [3,2,1]])

print(arr)
print(arr2)
arr3 = arr + arr2
print(arr3)

print(arr < arr2)

# broadcasting: make the shapes match
# by duplicating data
print(arr + np.array([5]))

print(arr + np.array([5,4,3]))


console.rule()

arr = np.array([4, 7, 9])
print("arr = ")
print(arr)
arr2 = np.array([2])
print("arr2 = ")
print(arr2)
print("arr2 broadcasted to arr shape =")
print(arr2 + np.zeros(arr.shape))
print("arr + arr2 = ")
print(arr + arr2)

console.rule()

arr = np.array([[4, 7, 9], [2, 3, 5]])
print("arr = ")
print(arr)
arr2 = np.array([2])
print("arr2 = ")
print(arr2)
print("arr2 broadcasted to arr shape =")
print(arr2 + np.zeros(arr.shape))
print("arr + arr2 = ")
print(arr + arr2)

console.rule()

arr = np.array([[4, 7, 9], [2, 3, 5]])
print("arr (shape %s) = " % str(arr.shape))
print(arr)
arr2 = np.array([[2], [3]])
print("arr2 (shape %s) = " % str(arr2.shape))
print(arr2)
print("arr2 broadcasted to arr shape =")
print(arr2 + np.zeros(arr.shape))
print("arr + arr2 = ")
print(arr + arr2)

console.rule()

arr = np.array([[4, 7, 9], [2, 3, 5]])
print("arr (shape %s) = " % str(arr.shape))
print(arr)
arr2 = np.array([[[2], [3]]])
print("arr2 (shape %s) = " % str(arr2.shape))
print(arr2)
print("arr2 broadcasted to arr (shape = %s) =" % str((arr2 + np.zeros(arr.shape)).shape))
print(arr2 + np.zeros(arr.shape))
print("arr + arr2 = ")
print(arr + arr2)

console.rule()

arr = np.array([[4, 7, 9], [2, 3, 5]])
print("arr (shape %s) = " % str(arr.shape))
print(arr)
arr2 = np.arange(2*1*2*1).reshape((2,1,2,1))
print("arr2 (shape %s) = " % str(arr2.shape))
print(arr2)
print("arr2 broadcasted to arr (shape = %s) =" % str((arr2 + np.zeros(arr.shape)).shape))
print(arr2 + np.zeros(arr.shape))
print("arr broadcasted to arr2 (shape = %s) =" % str((arr + np.zeros(arr2.shape)).shape))
print(arr + np.zeros(arr2.shape))
print("arr + arr2 = ")
print(arr + arr2)

console.rule()

print("Aggregate operations:")

print("arr = ")
print(arr)
print("arr.sum() = ")
print(arr.sum())
print("arr.sum(axis=0) = ")
print(arr.sum(axis=0))
print("arr.sum(axis=1) = ")
print(arr.sum(axis=-1))
print("arr.sum(axis=-1) = ")
print(arr.sum(axis=-1))
print("arr.sum(axis=-2) = ")
print(arr.sum(axis=-2))

console.rule()

arr = np.arange(2*3*5).reshape((2,3,5))
print("arr = ")
print(arr)
print("arr.mean() = ")
print(arr.mean())
print("arr.mean(axis=0) = ")
print(arr.mean(axis=0))
print("arr.mean(axis=1) = ")
print(arr.mean(axis=1))
print("arr.mean(axis=2) = ")
print(arr.mean(axis=2))
console.rule()

print("arr = ")
print(arr)
print("arr.min() = ")
print(arr.min())
print("arr.min(axis=0) = ")
print(arr.min(axis=0))
print("arr.min(axis=1) = ")
print(arr.min(axis=1))
print("arr.min(axis=2) = ")
print(arr.min(axis=2))
console.rule()

arr = np.random.rand(2,3,5)
print("arr = ")
print(arr)
print("arr.argmin() = ")
print(arr.argmin())
print("arr.argmin(axis=0) = ")
print(arr.argmin(axis=0))
print("arr.argmin(axis=1) = ")
print(arr.argmin(axis=1))
print("arr.argmin(axis=2) = ")
print(arr.argmin(axis=2))
console.rule()

# Examples for in-class Fri:
# Compute average score per student (all assignments)
# Compute average score per assignment (all students)
# Compute student minimum score, drop lowest assignment

