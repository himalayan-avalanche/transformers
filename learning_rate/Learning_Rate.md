# Whats Learning Rate?

Learning Rate is a crtical hyperpameter when it comes training deep learning models. Adaptive learning rates can help the parameters to converge faster to local minima optimal values.

##### PyTorch's optim.lr_scheduler
In PyTorch, optim.lr_scheduler is a module that provides various strategies for adjusting the learning rate during training. These learning rate schedulers allow you to change the learning rate of the optimizer at specific intervals, which can help the model converge better and faster, especially during the later stages of training.

> Note: Why Use lr_scheduler?

	•	Avoid Overshooting: High learning rates may cause the optimizer to overshoot the minimum, making it harder to converge. Reducing the learning rate can help the optimizer stabilize.
	•	Speed Up Convergence: Start with a higher learning rate to converge faster in the early stages and then reduce it as the model approaches the optimal solution.
	•	Fine-tuning: In some cases, after initial training, it’s beneficial to lower the learning rate to fine-tune the model’s performance.
 
#### Commonly Used Learning Rate Schedulers in PyTorch
1.	StepLR: Decreases the learning rate by a factor (gamma) every few epochs (based on a step size).
   
 ```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

 	•	step_size: Number of epochs after which the learning rate is decayed.
	•	gamma: Multiplicative factor for learning rate decay.

2.	MultiStepLR: Decreases the learning rate at specific epochs (milestones).
   
```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
```

