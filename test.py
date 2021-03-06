import torch
w = torch.rand(5)
w.requires_grad_()
print(w) 
s = w.sum() 
s.backward()
w.grad.zero_()
print(w.grad) # tensor([1., 1., 1., 1., 1.])
s.backward()
print(w.grad) # tensor([2., 2., 2., 2., 2.])
s.backward()
print(w.grad) # tensor([3., 3., 3., 3., 3.])
s.backward()
print(w.grad) # tensor([4., 4., 4., 4., 4.])
