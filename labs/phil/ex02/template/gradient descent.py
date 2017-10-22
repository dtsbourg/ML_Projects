# -*- coding: utf-8 -*-
"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    return -1/len(y) * np.transpose(tx).dot(e)


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w)
        grad = compute_gradient(y,tx,w)
        w = w - gamma * grad 
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws