import torch
import numpy as np

def arcpoint(x,a,b):
    # a and b are the endpoints of the arc,
    # returns true if x inside range of [a,b]
    bp = b % (2*np.pi)
    ap = a % (2*np.pi)
    ap = torch.where(ap>np.pi, ap - 2*np.pi, ap)
    bp = torch.where(bp>np.pi, bp - 2*np.pi, bp)
    #print('modulo')
    #print(ap)
    #print(bp)
    spread = torch.where(bp>ap, (bp-ap)/2, (bp+2*np.pi-ap)/2)
    #print('spread')
    #print(spread)
    inner = torch.where(spread<np.pi/2, True, False)
    x,a,b = thetaParse(x,a,b)
    #x = x% (2*np.pi)
    #a = a% (2*np.pi)
    #b = b% (2*np.pi)
    #print('arcpoint')
    #print(x)
    #print(a)
    #print(b)
    
    # a < x < b, b-a is smaller than pi
    one_in = torch.logical_and(torch.logical_and(a < x, x<b),(b-a)< np.pi)
    # b-a is larger than pi, x < a or x > b)
    two_in = torch.logical_and((b-a) > np.pi, torch.logical_or(x<a, x>b))
    
    # checking periodicity
    three_in = torch.logical_and(x < a, (x + 2*np.pi) < b)
    four_in = torch.logical_and(x > b, (x - 2*np.pi) > a)  


    # a < x < b, b-a is greater than pi
    one_out = torch.logical_and(torch.logical_and(x < b, (b-a)> np.pi),x-a>0)

    # b-a is larger than pi, x < a or x > b)
    two_out = torch.logical_and((b-a) < np.pi, torch.logical_or(x<a, x>b))
    # checking periodicity
    three_out = torch.logical_and(x < a, (x + 2*np.pi) < (b)) 
    four_out = torch.logical_and(x > b, (x - 2*np.pi) > (a))

    case1 = torch.logical_or(torch.logical_or(torch.logical_or(one_in, two_in), three_in),four_in)
    case2 = torch.logical_or(torch.logical_or(torch.logical_or(one_out, two_out), three_out),four_out)

    #case1 = torch.logical_or(one_in, two_in)
    #case2 = torch.logical_or(one_out, two_out)


    return torch.where(inner, case1, case2)
       
def pointDist(x,a,b):
    # x is point out of range
    # a is lower bound and b is upper bound
    # return true if closer to a, false otherwise
    x, a, b = thetaParse(x,a,b)
    x = x% (2*np.pi)
    a = a% (2*np.pi)
    b = b% (2*np.pi)

    x_a1 = torch.abs(x - a)
    x_b1 = torch.abs(x - b)
    x_a2 = 2*np.pi - x_a1
    x_b2 = 2*np.pi - x_b1

    x_a = torch.where(x_a1<x_a2, x_a1, x_a2)
    x_b = torch.where(x_b1<x_b2, x_b1, x_b2)

    
    return torch.where(x_a < x_b, True, False)


def thetaParse(x,a,b):
    #a = torch.where(a< -np.pi, a + 2*np.pi, a)
    #a = torch.where(a> np.pi, a - 2*np.pi, a)
    #b = torch.where(b< -np.pi, b + 2*np.pi, b)
    #b = torch.where(b> np.pi, b - 2*np.pi, b)
    x = x + 2*np.pi
    a = a + 2*np.pi
    b = b + 2*np.pi
    b = torch.where(b<a, b+2*np.pi, b)
    #temp = torch.clone(a)
    #a = torch.where(b<a, b,a)
    #b = torch.where(b==a, temp, b)
    
    #b = b - a
    #x = x - a
    return x, a, b
