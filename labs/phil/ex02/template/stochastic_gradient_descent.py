# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx.dot(w)
    return -1/len(y) * np.transpose(tx).dot(e)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = compute_loss(y,tx,w)
        grad = [0.0,0.0]

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, 1, True): # runs only once
            grad += compute_stoch_gradient(minibatch_y,minibatch_tx,w)
        
        w = w - gamma * grad 
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Stochastic Gradient Descent({bi}/{ti}): grad={g1},{g2}, loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, g1=grad[0], g2=grad[1], l=loss, w0=w[0], w1=w[1]))

    return losses, ws