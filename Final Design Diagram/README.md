Here is our system level architecture diagram

![Diagram](https://github.com/tzhang-Vincent/BatchLearning/blob/master/Final%20Design%20Diagram/Diagram.png)

Detailed Description:

We have three layers in our product: Data Access Layer, Business Logic Layer, Presentation Layer.

In the Data Access Layer, we use DAQ device get the motor data from the industrial motor we have including x-axis, y-axis vibration information and rotation speed, and implement the data preprocessing on them. 

In the Business Logic Layer, we apply batch training alogorithm which is Self-Organization Map algorithm to train the data at the motor edge and finally create a edge prediction model. When detecting the health condition of the specific motor, we use the model to calculate the Minimum Error Quantization of the algorithm output and make the prediction.What's more, the algorithm is embedded in Raspberry Pi.

In the Presentation Layer, we build a server in Python and create a connection between Raspberry Pi server and Web UI which is also developed by ourselves. Using javascripts, the real-time training progress could be shown on the web UI and the health condition of the motor could also be viewed.

Also, the Flow Chart Graph:

![FlowChart](https://github.com/tzhang-Vincent/BatchLearning/blob/master/Final%20Design%20Diagram/flow-chart.png)
