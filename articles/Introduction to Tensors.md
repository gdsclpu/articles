
![Learn TensorFlow for Machine Learning](https://github.com/prajwal3104/Articles/assets/70045720/03804302-3ff4-4f79-9f22-d9c63e692c5c)

SEO Title: "Tensor Basics: Types, Operations, and Applications in TensorFlow"

Meta Description: Learn the basics of tensors in TensorFlow. Explore tensor types, operations, and their applications in machine learning and data science. Perform mathematical computations, reshaping, and indexing. Discover the power of TensorFlow's tensors for efficient data manipulation and analysis.

## Introduction

In the exciting world of machine learning, artificial intelligence, and data science, there are fundamental concepts that form the building blocks of these fields. One such concept is tensors. If you're eager to dive into the world of Python, TensorFlow, and the incredible possibilities they offer, this article is dedicated to you. We will provide a detailed introduction to tensors, exploring their properties and how they are used in various applications.

But first, let's clarify what tensors actually are. Think of tensors as multi-dimensional arrays with a uniform type, which is known as a dtype. In TensorFlow, you can find a comprehensive list of supported dtypes at `tf.dtypes.DType`. If you are already familiar with NumPy, you'll find that tensors are somewhat similar to `np.arrays`.

Now, there's an important characteristic to keep in mind when dealing with tensors: they are immutable. Just like Python numbers and strings, you can't update the contents of a tensor once it is created. Instead, you create a new tensor with the desired modifications.

In the upcoming sections of this article, we will delve deeper into tensors, exploring their dimensions, operations, and their role in machine learning and artificial intelligence. By the end, you will have a solid understanding of what tensors are and how they can be utilized effectively in Python and TensorFlow.

So, let's begin this journey of exploring tensors and unlocking their potential in the world of data science and machine learning!

> **Note:** This is a series of articles sorted into categories. Each category is dedicated to a variety of topics related to TensorFlow, a powerful machine learning framework. The first part is an [introduction to TensorFlow](https://prajwaal.live/tensorflow-mastery-unlock-the-ml-power-potential)
, its benefits, and a compact guide or a overview to the framework.

### Basics of Tensors

To understand tensors better, let's start by creating some basic tensors.

The first type of tensor we'll look at is a "scalar" or "rank-0" tensor. A scalar tensor contains a single value and has no axes. In TensorFlow, we can create a scalar tensor using the `tf.constant()` function.

Here's an example code snippet that creates a rank-0 tensor with the value 4:

```python
import tensorflow as tf

# Create a rank-0 tensor (scalar)
rank_0_tensor = tf.constant(4)

# Print the tensor
print(rank_0_tensor)
```

Output:
```
tf.Tensor(4, shape=(), dtype=int32)
```

In the above code, we import the TensorFlow library and use the `tf.constant()` function to create a tensor with the value 4. The `tf.constant()` function creates an immutable tensor with the provided value. We then print the tensor using the `print()` function.

The output shows the created tensor with the value 4, its shape (empty parentheses indicate a scalar tensor), and its dtype (int32 in this case).

### Vectors: Rank-1 Tensors

In the realm of tensors, the next concept we'll explore is a "vector" or "rank-1" tensor. A vector can be thought of as a list of values and has a single axis.

Creating a vector in TensorFlow is straightforward. Let's take a look at an example:

```python
import tensorflow as tf

# Create a rank-1 tensor (vector)
rank_1_tensor = tf.constant([4.0, 6.0, 8.0])

# Print the tensor
print(rank_1_tensor)
```

Output:
```
tf.Tensor([4. 6. 8.], shape=(3,), dtype=float32)
```

In the code snippet above, we create a rank-1 tensor using the `tf.constant()` function. The values `[4.0, 6.0, 8.0]` form the elements of the vector. Similar to the previous example, we use the `print()` function to display the tensor.

The output shows the created tensor, which consists of the values 4.0, 6.0, and 8.0. The `shape` of the tensor is `(3,)`, indicating that it has three elements along a single axis. The `dtype` of the tensor is float32, as we explicitly specified the values to be floating-point numbers.

Understanding vectors is crucial in various applications, as they can represent features, data points, or mathematical quantities in machine learning and data science. By manipulating and operating on vectors, we can perform computations and derive meaningful insights from data.

### Matrices: Rank-2 Tensors

Now, let's delve into matrices, which are rank-2 tensors. A matrix is characterized by having two axes, forming a rectangular grid of values.

Creating a matrix in TensorFlow involves providing a 2D array of values. Here's an example:

```python
import tensorflow as tf

# Create a rank-2 tensor (matrix)
rank_2_tensor = tf.constant([[1, 4],
                             [2, 5],
                             [3, 6]], dtype=tf.float16)

# Print the tensor
print(rank_2_tensor)
```

Output:
```
tf.Tensor(
[[1. 4.]
 [2. 5.]
 [3. 6.]], shape=(3, 2), dtype=float16)
```

In the code above, we utilize the `tf.constant()` function to create a rank-2 tensor. The values are provided as a 2D array `[[1, 4], [2, 5], [3, 6]]`, where each row represents a list of elements. We also specify the `dtype` of the tensor as `tf.float16` to explicitly set the data type.

The output displays the created tensor, which consists of the provided values arranged in a grid-like structure. The `shape` of the tensor is `(3, 2)`, indicating that it has three rows and two columns. The `dtype` is float16, as specified during creation.

Matrices are extensively used in various mathematical operations, linear algebra, and machine learning algorithms. They serve as the foundation for representing datasets, performing matrix multiplications, and transforming data.

![tensor](https://github.com/gdsclpu/articles/assets/70045720/aa89128a-7f28-4e84-ba28-389ce59ae556)

### Tensors with Multiple Axes

In the world of tensors, it's important to recognize that tensors can have more than two axes. These tensors with higher dimensions offer a way to represent and process complex data structures.

Let's take a look at an example of a rank-3 tensor with three axes:

```python
import tensorflow as tf

# Create a rank-3 tensor
rank_3_tensor = tf.constant([
  [[5, 10, 15, 20, 25],
   [30, 35, 40, 45, 50]],
  [[55, 60, 65, 70, 75],
   [80, 85, 90, 95, 100]],
  [[105, 110, 115, 120, 125],
   [130, 135, 140, 145, 150]]
])

# Print the tensor
print(rank_3_tensor)
```

Output:
```
tf.Tensor(
[[[  5  10  15  20  25]
  [ 30  35  40  45  50]]

 [[ 55  60  65  70  75]
  [ 80  85  90  95 100]]

 [[105 110 115 120 125]
  [130 135 140 145 150]]], shape=(3, 2, 5), dtype=int32)
```

In the code snippet above, we create a rank-3 tensor using the `tf.constant()` function. The tensor is defined as a 3D array, with each element representing a list of values. The output showcases the tensor, displaying its values arranged across three axes. The `shape` of the tensor is `(3, 2, 5)`, indicating that it has three dimensions, two sub-dimensions, and five elements in each sub-dimension. The `dtype` of the tensor is `int32`.

Tensors with multiple axes are particularly useful in scenarios where we need to handle and manipulate complex data structures. They allow us to represent and process data in higher-dimensional spaces, providing valuable insights and facilitating advanced computations in machine learning and data science.

![tensor1](https://github.com/prajwal3104/Articles/assets/70045720/daf07dff-ad74-40d9-bc6a-17e97e603f67)

### Converting Tensors to NumPy Arrays

In TensorFlow, you have the flexibility to convert tensors to NumPy arrays, which can be useful when you want to leverage the functionality provided by the NumPy library. There are two common methods to perform this conversion: using `np.array()` from NumPy or the `numpy()` method available on tensors.

Let's take a look at an example that demonstrates both approaches using a rank-2 tensor:

```python
import tensorflow as tf
import numpy as np

# Create a rank-2 tensor
rank_2_tensor = tf.constant([[1, 4],
                             [2, 5],
                             [3, 6]], dtype=tf.float16)

# Convert tensor to NumPy array using np.array()
array_1 = np.array(rank_2_tensor)

# Convert tensor to NumPy array using tensor.numpy()
array_2 = rank_2_tensor.numpy()

# Print the arrays
print(array_1)
print(array_2)
```

Output:
```
array([[1., 4.],
       [2., 5.],
       [3., 6.]], dtype=float16)
```

In the code snippet above, we first create a rank-2 tensor using `tf.constant()`. We then demonstrate two methods of converting the tensor to a NumPy array. The first approach uses the `np.array()` function, which takes the tensor as input and returns a corresponding NumPy array. The second approach utilizes the `numpy()` method available on tensors, which directly converts the tensor to a NumPy array.

The output displays the NumPy arrays obtained from both conversion methods, showing the same result for both `array_1` and `array_2`.

Converting tensors to NumPy arrays allows you to seamlessly integrate TensorFlow with the rich ecosystem of tools and libraries available in the NumPy ecosystem. You can perform various numerical computations, statistical analysis, and visualization using NumPy functions on the converted arrays.

### Tensor Operations and Types

Tensors in TensorFlow are not limited to containing only floats and ints. They can also handle various other data types, including complex numbers and strings. Additionally, TensorFlow provides specialized tensor types to handle different shapes, such as ragged tensors and sparse tensors.

Let's explore some basic mathematical operations that can be performed on tensors, including addition, element-wise multiplication, and matrix multiplication:

```python
import tensorflow as tf

a = tf.constant([[5, 6],
                 [7, 8]])

b = tf.constant([[1, 1],
                 [1, 1]])

# Addition
print(tf.add(a, b), "\n")

# Element-wise multiplication
print(tf.multiply(a, b), "\n")

# Matrix multiplication
print(tf.matmul(a, b), "\n")
```

Output:
```
tf.Tensor(
[[6 7]
 [8 9]], shape=(2, 2), dtype=int32)

tf.Tensor(
[[5 6]
 [7 8]], shape=(2, 2), dtype=int32)

tf.Tensor(
[[11 11]
 [15 15]], shape=(2, 2), dtype=int32)
```

In the code snippet above, we create two tensors `a` and `b` using the `tf.constant()` function. We then perform various operations on these tensors. The `tf.add()` function adds the tensors element-wise, `tf.multiply()` performs element-wise multiplication, and `tf.matmul()` carries out matrix multiplication.

The output showcases the results of the operations, illustrating the element-wise addition, element-wise multiplication, and matrix multiplication.

Tensors and their operations play a fundamental role in various domains, including machine learning, deep learning, and numerical computing. They provide a flexible and efficient way to represent and manipulate data, enabling complex mathematical computations on multi-dimensional arrays.

### Tensor Operations: Beyond Basic Math

Tensors in TensorFlow are not only limited to basic mathematical operations like addition and multiplication. They are extensively used in a wide range of operations, often referred to as "Ops," that provide powerful functionalities for manipulating and analyzing data.

Let's explore a few examples of these tensor operations:

```python
import tensorflow as tf

c = tf.constant([[4.0, 11.0], [31.0, 8.0]])

# Find the largest value in the tensor
print(tf.reduce_max(c))

# Find the index of the largest value in the tensor
print(tf.math.argmax(c))

# Compute the softmax of the tensor
print(tf.nn.softmax(c))
```

Output:
```
tf.Tensor(31.0, shape=(), dtype=float32)
tf.Tensor([1 0], shape=(2,), dtype=int64)
tf.Tensor(
[[9.11051233e-04 9.99089003e-01]
 [1.00000000e+00 1.02618795e-10]], shape=(2, 2), dtype=float32)
```

In the code snippet above, we have a tensor `c` containing floating-point values. We apply different tensor operations to `c` to demonstrate their functionalities.

- `tf.reduce_max()` returns the largest value in the tensor.
- `tf.math.argmax()` returns the index of the largest value in the tensor.
- `tf.nn.softmax()` computes the softmax function on the tensor, which is commonly used in machine learning and classification tasks.

The output displays the results of these operations, showcasing the largest value, the index of the largest value, and the computed softmax probabilities.

TensorFlow provides a rich set of tensor operations that allow you to perform a variety of computations on tensors efficiently. These operations are vital in numerous applications, including machine learning, deep learning, data analysis, and scientific computing.

In the upcoming sections, we will continue exploring tensors, including tensor reshaping, indexing and slicing, and their applications in machine learning and data science. So, let's continue our exciting journey into the vast world of tensors!
