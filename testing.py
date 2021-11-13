
import numpy as np
import re

x = np.array([1, 2, 3, 4])
a = np.array([1, 2, 3, 4])
b = np.array([1, 1, 1, 1])
c = np.array([1, 1, 1, 1])
d = np.array([1, 1, 1, 1])

# # s_lists = [a, b, c, d]
# s_lists = [a, b]
# s = np.stack(s_lists)
# # x = x / sqrt(crossprod(x));
# # return(  as.vector((m %*% x) / sqrt(rowSums(m^2))) );
# x = x / np.sqrt(np.sum(x * x))
# print(x)
# r = s @ x
# print(r)
# den = np.sqrt(np.sum(np.square(s), axis=1))
# print(den)
# print(r / den)

# s_lists = [a, b, c, d]
# s = np.stack(s_lists)
# print(s)

for sent in re.split(r"\. *|\! *|\? *", "hello there! I like food. do you?"):
    if len(sent) > 0:
        print(sent)