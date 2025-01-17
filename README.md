# Neural Style Transfer

## Implementing Neural Style Transfer 
This project uses `VGG19` model trained on Imagenet.  
We will create one content model, and 3 style models   
out of intermediate layers of VGG19.  
You can also create 5 style models.  
<br>
#### Layers for Content Model
For Content Model, `block4_conv2` layer would be used.
```
content_model = model(inputs = vgg.input, outputs = vgg.get_layer('block4_conv2').output)
```
#### Layers for Style Model
For Style model we are using 3 models, but you can also use 5 models.   
The layers are:
```
style_layers = [
    'block1_conv1',
    'block3_conv1',
    'block5_conv1'
]
```
Creating different models:
```
style_models = [tf.keras.models.Model(inputs = model.inputs,
	outputs = model.get_layer(layer).output) for layer in style_layers]
```
### Creating variable for generated image
```
generated = tf.Variable(content, dtype = tf.float32)
```
### Function to calculate Content Cost
This function calculates the cost between the generated image and content image
```
def content_cost(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    cost = tf.reduce_mean(tf.square(a_C - a_G))
    return cost
```
### Function to calculate Style Cost
This function calculates the cost between the generated image and style image
```
style_model_weights = 1. / len(style_models)
def style_cost(style, generated):
    J_style = 0
    
    for style_model in style_models:
      	 a_S = style_model(style)
      	 a_G = style_model(generated)
      	 GS = gram_matrix(a_S)
      	 GG = gram_matrix(a_G)
      	 current_cost = tf.reduce_mean(tf.square(GS-GG))
      	 J_Style += current_cost*style_model_weights
   return J_style
```
Gram Matrix Function:
```
def gram_matrix(M):
    num_channels = tf.shape(M)[-1]
    M = tf.reshape(M, shape=(-1, num_channels))
    n = tf.shape(M)[0]
    G = tf.matmul(tf.transpose(M), M)
    return G / tf.cast(n, dtype=tf.float32)
```
### Training Loop
Our aim to decrease the overall cost ,i.e., style cost + content cost  
so, here's out trianing loop:
```
for i in range(iterations):
    iteration_time = time.time()
    with tf.GradientTape() as tape:
        J_content = content_cost(content, generated)
        J_style = content_cost(style, generated)
        J_total = alpha * J_content + beta * J_style
    grads = tape.gradient(J_total, generated)
    optimizer.apply_gradients([(grads, generated)])
```
## Thank you
Thanks to Coursera for this awesome course.  
And thanks to the instructor.  

Also, thanks to the tensorflow community who provided all the solutions to any questions that I had
## Improvements
The limitation of this algorithm is that it cannot create the image in real-time.   
You have to train the neural network to decrease the cost in order to create the mixed art.   
But a better method is available, which can be found at this website:[Real-time-neural-transfer](https://github.com/ChengBinJin/Real-time-style-transfer)  

## Implementing it Using Tensorflow JS
I am currently trying to port it to javascript with tensorflow.js  
I am new to TFJS, so, if you can help in anyway, then please head over to   
this repository [Deep-Learning-Gui-v1.0.1](https://github.com/prashant9316/Deep-Learning-GUI-beta/tree/nst)  
