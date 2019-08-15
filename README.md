# BatchLearning
Adaptive Motor Monitoring System Based on Machine Learning Batch Training

Project manager: Zhang Ting

Group member: Zhou Chenxin, Cheng Yiwei, Zhu Junru

Project Description:

1. University-enterprise cooperative project with CyberInsight Technology

2. Implemented batch training to realize abnormal detection for different motor within cyber-physical system

3. Realized edge online training on using Self-Organizing Map algorithm and motor vibration signal processing algorithm embedded in Raspberry Pi to overcome time-consuming and unsafe caused by traditional cloud one-time training

4. Realized server frame building in Python and its interaction with Web UI written in html and javascripts

Project Abstract:

Motor, as one of the most important equipment in various fields, plays an important role in today’s manufacturing. Effective motor health monitoring system can help detect the anomality in time or even detect the anomality in advance so that the production efficiency can be improved. A traditional data-driven approach is based on cloud training(transferring the data to the cloud and training the model on the cloud). However, that process is time-consuming and data may lose during that process. Moreover, traditional systems always adapt an “one-time” model, which means that the model will never change once the train finishes.

The monitoring system here is based on edge batch training and can solve the problem well. In the system, the process of train of model is place on the edge side (the side of motor). That reduces the time of transferring data and reduce the probability of loss of data. What’s more, as batch training supports training the model by batches of data, it guarantees the adaptivity of the model. That is because the users can update the model by a new batch of data based on the latest motor condition. By the way, the system provides a clear user interface where shows the real-time train process, the health score of the motor monitored and the historical data. Also, the user interface supports function such as “Retrain” the “Stoptrain” so the system can be applied in various real conditions.

![FinalDesign](https://github.com/tzhang-Vincent/BatchLearning/blob/master/EXPO.jpg)

And its system level diagram:

![Diagram](https://github.com/tzhang-Vincent/BatchLearning/blob/master/Final%20Design%20Diagram/Diagram.png)
