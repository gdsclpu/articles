![Learn TensorFlow for Machine Learning 2]([https://github.com/prajwal3104/Articles/assets/70045720/a17dceeb-8792-4b50-a205-695645ddd598](https://res.cloudinary.com/dckfb8ri8/image/upload/v1685191615/Blog/Featured%20Images/Learn_TensorFlow_for_Machine_Learning_2_cover_mn3rhl.png))


Meta Discrioption: Welcome back to our ongoing series, "Learn TensorFlow for Machine Learning," where we delve deeper into the fascinating world of ML using the tf framework.


## Introduction

Welcome back to our ongoing series, "Learn TensorFlow for Machine Learning," where we delve deeper into the fascinating world of machine learning using the TensorFlow framework. In our previous article, we introduced the fundamentals of TensorFlow and its importance in the field of artificial intelligence and machine learning. Today, we will continue our journey by exploring some more topics that will further enhance your understanding and skills in TensorFlow.

In the world of machine learning, artificial intelligence, and data science, TensorFlow has emerged as a powerful framework for building and training models. While TensorFlow can be used interactively like any Python library, it also offers additional tools for performance optimization and model export. One such tool is `tf.function`, which allows developers to separate pure-TensorFlow code from Python code, enabling enhanced performance and exportability.

## Understanding Graphs and `tf.function`

In TensorFlow, a graph represents a computational flow that defines the operations and dependencies between them. By constructing a graph, TensorFlow can optimize the execution of operations, making it more efficient than traditional Python execution. This optimization is particularly beneficial for complex and computationally intensive tasks commonly encountered in machine learning.

The `tf.function` decorator plays a crucial role in leveraging the benefits of graphs. When applied to a Python function, `tf.function` performs a process called tracing. During tracing, the function is executed in Python, but TensorFlow captures the computations and constructs an optimized graph representation of the operations performed within the function.

## Performance Optimization with `tf.function`

To illustrate the concept of graph construction and performance optimization, consider the following code snippet:

```python
@tf.function
def my_func(x):
  print('Tracing.\n')
  return tf.reduce_sum(x)
```

When you run this code for the first time, it doesn't produce any output. However, behind the scenes, TensorFlow traces the function and creates an optimized graph for future executions. Let's see the effect of this optimization by executing the function with a specific input:

```python
x = tf.constant([31, 1, 4])
my_func(x)
```

### Code Output

```
Tracing.
<tf.Tensor: shape=(), dtype=int32, numpy=36>
```

Here, the function is traced, and the optimized graph is utilized to compute the sum of the elements in the tensor `x`. It is important to note that the first execution involves the tracing process, which may take a bit longer due to graph construction. However, subsequent executions will benefit from the optimized graph, resulting in faster computation.

## Reusability and Signature Sensitivity

While the optimization provided by the `tf.function` is advantageous, it is crucial to understand that the generated graph may not be reusable for inputs with a different signature, such as varying shapes or data types. In such cases, TensorFlow generates a new graph specifically tailored to the input signature.

Let's modify our previous example and observe the behavior of `tf.function` when presented with a different input:

```python
x = tf.constant([8.0, 11.8, 14.3], dtype=tf.float32)
my_func(x)
```

### Code Output

```
Tracing.
<tf.Tensor: shape=(), dtype=float32, numpy=34.1>
```

As you can see, even though the function is the same, TensorFlow generates a new graph to accommodate the input's different data types and shapes. This signature sensitivity ensures accurate computation and prevents potential errors that could arise from incompatible inputs.

## Managing Variables and Functions in TensorFlow

### Modules and tf.Module

In TensorFlow, `tf.Module` is a class that allows you to manage `tf.Variable` objects and the corresponding `tf.function` objects. It facilitates two crucial functionalities:

1. **Variable Saving and Restoration**: You can save and restore the values of variables using `tf.train.Checkpoint`. This is especially useful during training to save and restore a model's state efficiently.

2. **Importing and Exporting**: With `tf.saved_model`, you can import and export the values of `tf.Variable` objects and their associated `tf.function` graphs. This enables running a model independently of the Python program that created it, enhancing its portability.

An example showcasing the usage of `tf.Module` is as follows:

```python
class MyModule(tf.Module):
  def __init__(self, value):
    self.weight = tf.Variable(value)

  @tf.function
  def multiply(self, x):
    return x * self.weight
```

Here, we define a subclass of `tf.Module` named `MyModule`, which includes a variable `weight` and a `tf.function` called `multiply`. The `multiply` function performs element-wise multiplication between the input `x` and the `weight` variable.

Let's see the code in action:

```python
mod = MyModule(3)
mod.multiply(tf.constant([5, 6, 3]))
```

### Code Output

```
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([15, 18,  9], dtype=int32)>
```

In this example, we instantiate `MyModule` with an initial value of 3 and invoke the `multiply` function on a tensor. The output represents the element-wise multiplication between the tensor and the `weight` variable.

### Saving and Loading Modules

Once you have defined and utilized a `tf.Module` object, you can save it for future use or deployment. To save the module, you can employ `tf.saved_model.save()` and specify the desired save path:

```python
save_path = './saved'
tf.saved_model.save(mod, save_path)
```

The resulting SavedModel is independent of the code that created it. It can be loaded from Python, other language bindings, TensorFlow Serving, or even converted to run with TensorFlow Lite or TensorFlow.js.

To load and utilize the saved module, you can use `tf.saved_model.load()`:

```python
reloaded = tf.saved_model.load(save_path)
reloaded.multiply(tf.constant([5, 6, 3]))
```
### Code Output

```
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([15, 18,  9], dtype=int32)>
```

### Layers and Models: Building and Training Complex Models

Built on top of `tf.Module`, TensorFlow provides two higher-level classes: `tf.keras.layers.Layer` and `tf.keras.Model`. These classes offer additional functionality and convenience methods for constructing, training, and saving complex models.

`tf.keras.layers.Layer` serves as a base class for implementing custom layers. It provides a structure to define layers with trainable weights and can be used to build custom architectures or extend existing ones.

Extending `tf.keras.layers.Layer`, `tf.keras.Model` adds methods for training, evaluation, and model saving. It is commonly used as a container for multiple layers, enabling the definition and management of complex neural network architectures.

By leveraging the capabilities of `tf.keras.layers.Layer` and `tf.keras.Model`, you can create advanced models, train them on extensive datasets, and save their configurations and weights for future use or deployment.

- In conclusion: The concepts of modules, layers, and models in TensorFlow are crucial for effectively managing variables and functions. They provide mechanisms for saving and restoring variable values, exporting models for deployment, and constructing complex architectures. Utilizing these features empowers you to develop powerful and portable machine learning models that can be easily shared, reused, and deployed across different environments.


## Let's Build a Model: Training a Basic Model from Scratch

Now, we will put together the concepts of Tensors, Variables, Automatic Differentiation, Graphs, tf.function, Modules, Layers, and Models to build a basic machine learning model from scratch using TensorFlow. This part is dedicated to anyone who wants to learn and understand the process of building and training a model. 

> **Note:** Remember as said earlier this and the previous articles are just an overview, let us dig deeper into each topics in coming articles.

### Generating Example Data

To begin, let's create some example data. We will generate a cloud of points that roughly follows a quadratic curve. We'll use the Matplotlib library to visualize the data.

```python
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['figure.figsize'] = [10, 6]

x = tf.linspace(-3, 3, 201)
x = tf.cast(x, tf.float32)

def f(x):
  y = x**2 + 2*x - 8
  return y

y = f(x) + tf.random.normal(shape=[201])

plt.plot(x.numpy(), y.numpy(), '.', label='Data')
plt.plot(x, f(x), label='Ground truth')
plt.legend();
```
### Code Output
![1](https://res.cloudinary.com/dckfb8ri8/image/upload/v1685191588/Blog/Blog%20Images/learn%20tensorflow%202/1_ytrc8g.png)


### Creating a Quadratic Model

Next, we'll define a quadratic model with randomly initialized weights and a bias. The model will take an input `x` and predict the corresponding output using the equation `y = w_q * x^2 + w_l * x + b`, where `w_q` represents the weight for the quadratic term, `w_l` represents the weight for the linear term, and `b` represents the bias term.

```python
class Model(tf.Module):
  def __init__(self):
    rand_init = tf.random.uniform(shape=[3], minval=0., maxval=5., seed=22)
    self.w_q = tf.Variable(rand_init[0])
    self.w_l = tf.Variable(rand_init[1])
    self.b = tf.Variable(rand_init[2])

  @tf.function
  def __call__(self, x):
    return self.w_q * (x**2) + self.w_l * x + self.b

quad_model = Model()
```

We also define a function `plot_preds` that helps visualize the model predictions.

```python
def plot_preds(x, y, f, model, title):
  plt.figure()
  plt.plot(x, y, '.', label='Data')
  plt.plot(x, f(x), label='Ground truth')
  plt.plot(x, model(x), label='Predictions')
  plt.title(title)
  plt.legend()
```

Let's plot the initial predictions of the model:

```python
plot_preds(x, y, f, quad_model, 'Initial Predictions')
```
### Code Output
![2](https://res.cloudinary.com/dckfb8ri8/image/upload/v1685191589/Blog/Blog%20Images/learn%20tensorflow%202/2_lu1njs.png)

### Defining the Loss Function

Since the model aims to predict continuous values, we will use the mean squared error (MSE) as the loss function. The MSE calculates the mean of the squared differences between the predicted values and the ground truth.

```python
def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred - y))
```

### Training the Model

We will now write a basic training loop to train the model from scratch. The loop will use the MSE loss function and its gradients with respect to the input to update the model's parameters. We will use mini-batches for training, which provides memory efficiency and faster convergence. The `tf.data.Dataset` API is used for batching and shuffling the data.

```python
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=x.shape[0]).batch(batch_size)


epochs = 150
learning_rate = 0.01
losses = []

for epoch in range(epochs):
  for x_batch, y_batch in dataset:
    with tf.GradientTape() as tape:
      batch_loss = mse_loss(quad_model(x_batch), y_batch)
    grads = tape.gradient(batch_loss, quad_model.variables)
    for g,v in zip(grads, quad_model.variables):
        v.assign_sub(learning_rate*g)
  
  loss = mse_loss(quad_model(x), y)
  losses.append(loss)
  if epoch % 10 == 0:
    print(f'Mean squared error for step {epoch}: {loss.numpy():0.3f}')

plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title('MSE loss vs training iterations')
```

### Code Output
```
Mean squared error for step 0: 35.056
Mean squared error for step 10: 11.213
Mean squared error for step 20: 4.120
Mean squared error for step 30: 1.982
Mean squared error for step 40: 1.340
Mean squared error for step 50: 1.165
Mean squared error for step 60: 1.118
Mean squared error for step 70: 1.127
Mean squared error for step 80: 1.087
Mean squared error for step 90: 1.087
Mean squared error for step 100: 1.098
Mean squared error for step 110: 1.094
Mean squared error for step 120: 1.086
Mean squared error for step 130: 1.089
Mean squared error for step 140: 1.088
```

![3](https://res.cloudinary.com/dckfb8ri8/image/upload/v1685191588/Blog/Blog%20Images/learn%20tensorflow%202/3_u3v936.png)


### Evaluating the Trained Model

Let's observe the performance of the trained model:

```python
plot_preds(x, y, f, quad_model, 'Predictions after Training')
```
### Code Output
![4](https://res.cloudinary.com/dckfb8ri8/image/upload/v1685191589/Blog/Blog%20Images/learn%20tensorflow%202/4_oyzlpe.png)

### Utilizing tf.keras for Training

While implementing a training loop from scratch is educational, TensorFlow's `tf.keras` module provides convenient utilities for training models. We can use the `Model.compile` and `Model.fit` methods to simplify the training process.

To demonstrate this, we'll create a Sequential model using `tf.keras.Sequential`. We'll use the dense layer (`tf.keras.layers.Dense`) to learn linear relationships and a lambda layer (`tf.keras.layers.Lambda`) to transform the input for capturing the quadratic relationship.

```python
new_model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.stack([x, x**2], axis=1)),
    tf.keras.layers.Dense(units=1, kernel_initializer=tf.random.normal)
])

new_model.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
)

history = new_model.fit(x, y,
                        epochs=150,
                        batch_size=32,
                        verbose=0)

new_model.save('./my_new_model')
```

Let's visualize the training progress:

```python
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylim([0, max(plt.ylim())])
plt.ylabel('Loss [Mean Squared Error]')
plt.title('Keras training progress')
```
### Code Output
![5](https://res.cloudinary.com/dckfb8ri8/image/upload/v1685191589/Blog/Blog%20Images/learn%20tensorflow%202/5_seqhbm.png)

Finally, we can evaluate the performance of the Keras model:

```python
plot_preds(x, y, f, new_model, 'Predictions after Training')
```
### Code Output
![6](https://res.cloudinary.com/dckfb8ri8/image/upload/v1685191589/Blog/Blog%20Images/learn%20tensorflow%202/6_k8j0dt.png)

### Conclusion

In this article, we built a basic machine learning model from scratch using TensorFlow. We started by generating example data, created a quadratic model, defined a loss function, and trained the model using a training loop. We also explored utilizing the `tf.keras` module for a more convenient training process. By understanding these concepts, you can now begin building and training your own machine-learning models using TensorFlow.

This is not the end of this series, this was just an overview of how we can make use of the `TensorFlow` framework to ease our model-building in Machine Learning. In upcoming articles, we will dive deeper into every topic break them into parts, and clearly understand.
